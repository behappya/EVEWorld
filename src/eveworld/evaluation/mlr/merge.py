"""Alignment of detections and masks with the sampled latent timeline.

The protocol evaluates a clip on a fixed grid of :data:`DEFAULT_FRAME_COUNT` sampled timestamps
(the WorldArena ``sample_mode="round"`` grid), while the detectors and the annotation of the
reference clip speak in their own timelines: Grounding-DINO and SAM2 can run on every source
frame, and the IGR annotation of §8.3 is indexed by latent timestamp. This module joins the two.

:func:`align_timestamps` returns the source-frame index of every sampled timestamp. The rule is
the one of :func:`eveworld.data.transforms.video.sample_indices`, imported at module scope, with
a local copy of it as a fallback so that this module also imports where the data pipeline's
optional dependencies are missing. :func:`merge_detections` and :func:`merge_track_masks` map
per-frame detections and ``(T, N, H, W)`` masks onto that grid by nearest source frame.
:func:`build_metadata` packs the timeline, the counts the annotation expects and the annotated
target cells into the per-clip metadata, and :func:`protocol_metadata` records the settings a run
used, so that a result file can be audited without the code that produced it.

Array shapes: ``sample_indices`` is ``(S,)`` int, :func:`merge_track_masks` takes ``(T, N, H, W)``
masks and returns ``(S, N, H, W)`` with the same dtype, :func:`build_metadata` returns its
``timestamps`` as ``(S,)`` int64 and its ``expected_counts`` and ``instances`` as ``(S,)`` lists.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from eveworld.evaluation.mlr.detector import expected_count
from eveworld.utils.logging import get_logger

try:  # the transform module pulls in torch, which a numpy-only environment does not have
    from eveworld.data.transforms import video as _video
except ImportError:  # pragma: no cover - depends on the installed extras
    _video = None

logger = get_logger(__name__)

__all__ = [
    "DEFAULT_FRAME_COUNT",
    "DEFAULT_SAMPLE_MODE",
    "MLR_DEFAULTS",
    "align_timestamps",
    "build_metadata",
    "merge_detections",
    "merge_track_masks",
    "protocol_metadata",
]

DEFAULT_FRAME_COUNT = 24
DEFAULT_SAMPLE_MODE = "round"
SAMPLE_MODES = ("linspace", "round", "uniform")

# The settings the WorldArena protocol ran with, under the names the configuration files use.
MLR_DEFAULTS: dict[str, Any] = {
    "deviation_mode": "symmetric",
    "occlusion_rule": "paper_overlap",
    "sample_mode": DEFAULT_SAMPLE_MODE,
    "frame_count": DEFAULT_FRAME_COUNT,
    "persistence": 2,
    "tau_occ": 0.15,
    "reliable_logit": 0.0,
    "reliable_area_ratio": 0.50,
    "presence_logit": 0.0,
    "contact_margin_ratio": 0.035,
    "object_topk": 6,
    "object_box_threshold": 0.35,
    "robot_prompt": "robot gripper",
    "robot_topk": 3,
    "robot_box_threshold": 0.15,
    "gripper_overlap_threshold": 0.35,
    "min_center_distance": 60.0,
    "initial_count_source": "filtered",
    "sampled_count_source": "filtered",
    "device": "cuda",
    "profile_name": None,
    "gdino_path": None,
    "sam2_checkpoint": None,
    "sam2_model_config": "configs/sam2.1/sam2.1_hiera_t.yaml",
}

if _video is None:  # the fallback below repeats the same rule, but the delegation is preferred
    logger.debug("eveworld.data.transforms.video is unavailable; using the local sampling rule")


def _local_sample_indices(num_frames: int, num_out: int, mode: str) -> np.ndarray:
    """The sampling rule of the data pipeline, for environments where it cannot be imported."""
    if mode not in SAMPLE_MODES:
        raise ValueError(f"Unknown sample mode: {mode!r}; expected one of {SAMPLE_MODES}")
    frames = int(num_frames)
    count = int(num_out)
    if frames <= 0:
        return np.zeros(0, dtype=np.int64)
    if count <= 1:
        return np.array([0], dtype=np.int64)
    last = frames - 1
    if mode in ("linspace", "uniform"):
        indices = np.linspace(0, last, count).astype(np.int64)
    else:
        steps = np.arange(count, dtype=np.float64) * last / (count - 1)
        indices = np.floor(steps + 0.5).astype(np.int64)
    return np.clip(indices, 0, last).astype(np.int64)


def _sample_frame_indices(num_frames: int, num_out: int, mode: str) -> np.ndarray:
    """Sampled indices from the data pipeline, or from the local copy of its rule."""
    delegate = None if _video is None else getattr(_video, "sample_indices", None)
    if delegate is None:
        return _local_sample_indices(num_frames, num_out, mode)
    return np.asarray(delegate(int(num_frames), int(num_out), mode), dtype=np.int64)


def align_timestamps(
    frame_count: int,
    sample_count: int = DEFAULT_FRAME_COUNT,
    mode: str = DEFAULT_SAMPLE_MODE,
) -> np.ndarray:
    """Source-frame index of every sampled timestamp of a clip.

    Timestamp ``i`` reads source frame ``round(i * (F - 1) / (S - 1))`` under the paper protocol
    ``mode="round"``; ``"linspace"`` and its alias ``"uniform"`` keep the truncated
    ``numpy.linspace`` behaviour of the latent extraction scripts. The rule is the one of
    :func:`eveworld.data.transforms.video.sample_indices`; it is repeated locally only when that
    module cannot be imported, so both paths return the same indices.

    Args:
        frame_count: number of frames in the source clip.
        sample_count: sampled timestamps wanted; 24 is the protocol default.
        mode: one of :data:`SAMPLE_MODES`.

    Returns:
        ``(sample_count,)`` int64 source-frame indices within ``[0, frame_count - 1]``, ascending
        but holding repeats once ``sample_count`` exceeds ``frame_count``. A clip with no frames
        gives an empty array, and ``sample_count <= 1`` gives the single index ``0``.
    """
    return _sample_frame_indices(frame_count, sample_count, mode)


def _wanted_indices(sample_indices: Any) -> np.ndarray:
    """Sampled source-frame indices as a flat int64 array."""
    return np.asarray(sample_indices, dtype=np.int64).reshape(-1)


def _nearest_positions(available: np.ndarray, wanted: np.ndarray) -> np.ndarray:
    """Positions in the ascending ``available`` indices nearest to each wanted index.

    Ties go to the earlier entry, which keeps the mapping deterministic: a timestamp exactly
    between two frames reads the frame before it.
    """
    if available.size == 1:
        return np.zeros(wanted.shape, dtype=np.int64)
    positions = np.searchsorted(available, wanted, side="left")
    high = np.clip(positions, 1, available.size - 1)
    low = high - 1
    take_low = (wanted - available[low]) <= (available[high] - wanted)
    return np.where(take_low, low, high).astype(np.int64)


def _sorted_frames(per_frame: Any) -> tuple[np.ndarray, list[Any]]:
    """Per-frame detections as ascending frame indices plus their payloads."""
    if per_frame is None:
        return np.zeros(0, dtype=np.int64), []
    if isinstance(per_frame, Mapping):
        items = sorted(per_frame.items(), key=lambda item: int(item[0]))
        keys = np.asarray([int(key) for key, _ in items], dtype=np.int64)
        return keys, [value for _, value in items]
    values = list(per_frame)
    return np.arange(len(values), dtype=np.int64), values


def _detection_list(value: Any) -> list[Any]:
    """One frame's detections as a list; ``None`` is no detection and a bare one is wrapped."""
    if value is None:
        return []
    if isinstance(value, Mapping):
        return [value]
    if isinstance(value, np.ndarray):
        return list(value) if value.ndim > 0 else [value.item()]
    if isinstance(value, (str, bytes)):
        return [value]
    if isinstance(value, (list, tuple)):
        return list(value)
    if hasattr(value, "__iter__"):
        return list(value)
    return [value]


def merge_detections(per_frame: Any, sample_indices: Any) -> list[list[Any]]:
    """Per-frame detections assigned to the sampled latent timeline.

    The detector may run on every source frame while the latent timeline holds only the sampled
    timestamps the annotation is indexed by. Each sampled timestamp takes the detections of the
    source frame nearest to it, ties going to the earlier frame, so every timestamp is backed by
    real detections and none is interpolated into existence.

    Args:
        per_frame: one clip's detections, either a mapping from source-frame index to that frame's
            detections or a sequence indexed from ``0``. A frame without detections is an empty
            sequence or ``None``; a single detection that is not a sequence is wrapped in a list.
        sample_indices: ``(S,)`` source-frame indices, for example from :func:`align_timestamps`.

    Returns:
        ``(S,)`` list of lists, one list of detections per sampled timestamp; an empty input gives
        ``S`` empty lists rather than an error.
    """
    keys, values = _sorted_frames(per_frame)
    wanted = _wanted_indices(sample_indices)
    if keys.size == 0:
        return [[] for _ in range(int(wanted.size))]
    positions = _nearest_positions(keys, wanted)
    return [_detection_list(values[int(position)]) for position in positions]


def merge_track_masks(masks: Any, sample_indices: Any) -> np.ndarray:
    """Tracking masks of a clip resampled onto the sampled latent timeline.

    A mask stack such as the SAM2 output has one entry per source frame, whereas the counting rule
    and the occlusion table are indexed by sampled timestamp; the two timelines are matched by
    nearest source frame, with the same tie rule as :func:`merge_detections`.

    Args:
        masks: ``(T, N, H, W)`` masks of the clip, boolean or numeric, with ``T >= 1``.
        sample_indices: ``(S,)`` source-frame indices, for example from :func:`align_timestamps`.

    Returns:
        ``(S, N, H, W)`` masks of the nearest frames, with the dtype of ``masks``.
    """
    array = np.asarray(masks)
    if array.ndim != 4:
        raise ValueError(f"masks must have shape (T, N, H, W), got {array.shape}")
    if array.shape[0] == 0:
        raise ValueError("masks must hold at least one frame")
    wanted = _wanted_indices(sample_indices)
    positions = _nearest_positions(np.arange(array.shape[0], dtype=np.int64), wanted)
    return np.ascontiguousarray(array[positions])


def _length_of(value: Any) -> int | None:
    """Length of a timeline, or ``None`` when ``value`` does not describe one."""
    if value is None or isinstance(value, (str, bytes)):
        return None
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return int(value)
    try:
        return len(value)
    except TypeError:
        return None


def _frame_total(frames: Any, fallback: int) -> int:
    """Number of source frames in ``frames``, which may be a clip, a count or nothing."""
    if frames is None:
        return int(fallback)
    if isinstance(frames, (int, np.integer)) and not isinstance(frames, bool):
        return int(frames)
    shape = getattr(frames, "shape", None)
    if shape is not None and len(shape) > 0:
        return int(shape[0])
    length = _length_of(frames)
    return int(fallback) if length is None else length


def _as_count(value: Any) -> int | None:
    """Expected count as an integer, or ``None`` when the annotation does not state one."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _instance_cells(per_frame: Any, count: int) -> list[list[tuple[int, int]]]:
    """Annotated ``(row, col)`` target cells of every sampled timestamp."""
    instances: list[list[tuple[int, int]]] = []
    for index in range(count):
        frame = None
        if per_frame is not None:
            if isinstance(per_frame, Mapping):
                frame = per_frame.get(index, per_frame.get(str(index)))
            else:
                try:
                    frame = per_frame[index]
                except (IndexError, KeyError, TypeError):
                    frame = None
        cells: list[tuple[int, int]] = []
        if isinstance(frame, Mapping):
            target = frame.get("target_cell", frame.get("target"))
            if target is not None:
                values = np.asarray(target, dtype=np.int64).reshape(-1)
                if values.size >= 2:
                    cells.append((int(values[0]), int(values[1])))
        instances.append(cells)
    return instances


def build_metadata(
    annotation: Mapping[str, Any] | None,
    frames: Any = None,
    sample_count: int = DEFAULT_FRAME_COUNT,
    *,
    mode: str = DEFAULT_SAMPLE_MODE,
) -> dict[str, Any]:
    """Per-clip metadata describing the sampled timeline of a clip.

    The metadata answers the three questions the instrument asks of a clip: where every sampled
    timestamp sits in the source video, how many target instances the annotation expects to be
    visible there, and where the annotation puts the target. The counts come from
    :func:`eveworld.evaluation.mlr.detector.expected_count`, so a timestamp whose count the
    annotation does not constrain is reported as ``None`` instead of a made-up number.

    Args:
        annotation: IGR annotation of the reference clip (§8.3 schema: ``per_lat_frame``,
            ``inventory_cells``, ``n_inventory``), or the per-clip metadata that embeds it. A
            mapping, or ``None`` for a timeline with no annotation.
        frames: source clip, its frame count, or anything with a leading frame axis; it supplies
            the frame total :func:`align_timestamps` samples. When absent, the annotation's own
            ``timestamps`` or ``num_frames`` are used, and a clip whose timeline is its source
            video falls back to the identity grid.
        sample_count: sampled timestamps of the timeline when the annotation does not fix one.
        mode: sampling mode, see :func:`align_timestamps`.

    Returns:
        A mapping holding exactly

        * ``expected_counts``: ``(S,)`` list of ``int | None``, one per sampled timestamp;
        * ``timestamps``: ``(S,)`` int64 source-frame indices;
        * ``instances``: ``(S,)`` list of lists of ``(row, col)`` target cells, empty where the
          annotation lost the target, so a caller can tell a missing target from an empty clip;
        * ``sample_mode``: the sampling mode used;
        * ``frame_count``: ``S``, the number of sampled timestamps.
    """
    mapping: dict[str, Any] = dict(annotation) if isinstance(annotation, Mapping) else {}
    per_frame = mapping.get("per_lat_frame")
    if isinstance(per_frame, Mapping):
        per_frame = [per_frame.get(index, per_frame.get(str(index))) for index in range(len(per_frame))]
        mapping["per_lat_frame"] = per_frame

    count = _length_of(per_frame)
    if count is None:
        count = _length_of(mapping.get("timestamps"))
    if count is None:
        count = _length_of(mapping.get("expected_counts"))
    if count is None:
        count = _length_of(mapping.get("n_lat"))
    if count is None:
        count = int(sample_count)
    count = max(0, int(count))

    total = _frame_total(frames, _length_of(mapping.get("num_frames")) or count)
    provided_timeline = mapping.get("timestamps") if frames is None else None
    if count == 0:
        timestamps = np.zeros(0, dtype=np.int64)
    elif provided_timeline is not None and _length_of(provided_timeline) == count:
        timestamps = np.asarray(provided_timeline, dtype=np.int64).reshape(-1)
    else:
        timestamps = align_timestamps(total, count, mode)

    if per_frame is not None:
        expected_counts = [expected_count(mapping, index) for index in range(count)]
    else:
        provided_counts = mapping.get("expected_counts")
        if isinstance(provided_counts, Mapping):
            expected_counts = [_as_count(provided_counts.get(str(int(stamp)), provided_counts.get(int(stamp)))) for stamp in timestamps]
        elif provided_counts is not None and _length_of(provided_counts):
            values = list(provided_counts)
            expected_counts = [_as_count(values[index]) for index in range(count)]
        else:
            expected_counts = [None] * count

    return {
        "expected_counts": expected_counts,
        "timestamps": timestamps,
        "instances": _instance_cells(per_frame, count),
        "sample_mode": mode,
        "frame_count": count,
    }


def _setting(settings: Any, *names: str, default: Any = None) -> Any:
    """First entry of ``settings`` among ``names``, or ``default``."""
    if settings is None:
        return default
    for name in names:
        value = _setting_entry(settings, name)
        if value is not None:
            return value
    return default


def _setting_entry(settings: Any, name: str) -> Any:
    """One named entry of a mapping, an OmegaConf object or an argparse namespace."""
    if isinstance(settings, Mapping):
        return settings.get(name)
    getter = getattr(settings, "get", None)
    if callable(getter):
        try:
            value = getter(name, None)
        except TypeError:
            value = None
        if value is not None:
            return value
    return getattr(settings, name, None)


def _value(settings: Any, key: str, *aliases: str, cast: Any = None) -> Any:
    """One protocol setting, defaulting to :data:`MLR_DEFAULTS` and cast to its own kind."""
    default = MLR_DEFAULTS[key]
    value = _setting(settings, key, *aliases, default=default)
    if cast is None:
        return value
    try:
        return cast(value)
    except (TypeError, ValueError):
        return cast(default)


def protocol_metadata(settings: Any = None, initial_count: int | None = None) -> dict[str, Any]:
    """Audit block recording the settings a protocol run used.

    The block is written next to every result so that a table can be recomputed from the files
    alone. Its keys are the ones the WorldArena protocol recorded, including the audit names
    ``persistence_samples`` and ``object_prompt_topk`` of the ``persistence`` and ``object_topk``
    settings; a value the settings do not carry falls back to :data:`MLR_DEFAULTS`, so a partly
    specified profile still produces a complete block.

    Args:
        settings: mapping (``dict``, OmegaConf), argparse namespace or ``None`` for the defaults.
        initial_count: reference inventory ``N_0`` of the clip, when it is known.

    Returns:
        The audit mapping, ``{"protocol": "mlr_occlusion_v1", ...}``, holding the protocol name,
        the sampling and counting settings, the initial count and its source, the detector and
        SAM2 settings, and the profile name when the settings carry one.
    """
    return {
        "protocol": "mlr_occlusion_v1",
        "deviation_mode": _value(settings, "deviation_mode"),
        "occlusion_rule": _value(settings, "occlusion_rule"),
        "persistence_samples": _value(settings, "persistence", "persistence_samples", "k", cast=int),
        "tau_occ": _value(settings, "tau_occ", cast=float),
        "reliable_logit": _value(settings, "reliable_logit", cast=float),
        "reliable_area_ratio": _value(settings, "reliable_area_ratio", cast=float),
        "presence_logit": _value(settings, "presence_logit", cast=float),
        "contact_margin_ratio": _value(settings, "contact_margin_ratio", cast=float),
        "sample_mode": _value(settings, "sample_mode"),
        "frame_count": _value(settings, "frame_count", "sample_count", cast=int),
        "initial_count": None if initial_count is None else int(initial_count),
        "initial_count_source": _value(settings, "initial_count_source"),
        "sampled_count_source": _value(settings, "sampled_count_source"),
        "object_prompt_topk": _value(settings, "object_topk", "object_prompt_topk", cast=int),
        "object_box_threshold": _value(settings, "object_box_threshold", cast=float),
        "robot_prompt": _value(settings, "robot_prompt"),
        "robot_topk": _value(settings, "robot_topk", cast=int),
        "robot_box_threshold": _value(settings, "robot_box_threshold", cast=float),
        "gripper_overlap_threshold": _value(settings, "gripper_overlap_threshold", cast=float),
        "min_center_distance": _value(settings, "min_center_distance", cast=float),
        "sam2_checkpoint": _value(settings, "sam2_checkpoint"),
        "sam2_model_config": _value(settings, "sam2_model_config"),
        "device": _value(settings, "device"),
        "profile": _value(settings, "profile_name", "profile"),
    }
