"""Open-vocabulary detection with GroundingDINO.

The detector takes a text prompt such as ``"small red cube. grey table."`` and returns the boxes of the
matching objects in every frame. GroundingDINO itself is imported lazily so that the prompt helpers and
the dataclass can be used on machines that only run inference elsewhere.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from eveworld.data.parsers.instruction_parser import VERBS, parse_instruction
from eveworld.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = ["Detection", "GroundingDinoDetector", "box_area_ratio", "parse_objects"]

CONFIG_ENV_VAR = "GROUNDING_DINO_CONFIG"
WEIGHTS_ENV_VAR = "GROUNDING_DINO_WEIGHTS"

_WORD_RE = re.compile(r"[a-z']+")


@dataclass
class Detection:
    """A single open-vocabulary detection.

    Attributes:
        box: ``(4,)`` float32 array of pixel coordinates in ``xyxy`` order.
        score: Detection confidence in ``[0, 1]``.
        label: Text phrase that matched the detected region.
    """

    box: np.ndarray
    score: float
    label: str

    def __post_init__(self) -> None:
        self.box = np.asarray(self.box, dtype=np.float32).reshape(4)
        self.score = float(self.score)
        self.label = str(self.label)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly view of the detection.

        Returns:
            Dictionary with ``box`` as a list of four floats, ``score`` and ``label``.
        """
        return {"box": [float(v) for v in self.box], "score": self.score, "label": self.label}


def parse_objects(prompt: str) -> list[str]:
    """Split a grounding prompt into individual object phrases.

    Phrases are separated by periods, empty entries are dropped and duplicates are removed while keeping
    the order of first appearance. A single phrase that contains a manipulation verb is passed through
    :func:`eveworld.data.parsers.instruction_parser.parse_instruction` so that captions taken verbatim
    from an instruction still yield bare object names.

    Args:
        prompt: Free-form text, e.g. ``"small red cube. grey table."`` or ``"Pick up the bowl."``.

    Returns:
        List of distinct object phrases.
    """
    phrases: list[str] = []
    for part in str(prompt).split("."):
        phrase = part.strip()
        if phrase and phrase not in phrases:
            phrases.append(phrase)
    if len(phrases) == 1:
        words = set(_WORD_RE.findall(phrases[0].lower()))
        if words & set(VERBS):
            objects = parse_instruction(phrases[0]).objects
            if objects:
                return list(objects)
    return phrases


def box_area_ratio(box: np.ndarray, shape: Sequence[int]) -> float:
    """Return the fraction of the frame covered by ``box``.

    Args:
        box: ``(4,)`` array in pixel ``xyxy`` order; may extend beyond the frame.
        shape: Frame shape whose first two entries are ``(height, width)``.

    Returns:
        Intersection area with the frame divided by the frame area, in ``[0, 1]``.
    """
    height, width = int(shape[0]), int(shape[1])
    if height <= 0 or width <= 0:
        return 0.0
    x0, y0, x1, y1 = (float(v) for v in np.asarray(box, dtype=np.float32).reshape(4))
    left = max(0.0, min(x0, x1))
    top = max(0.0, min(y0, y1))
    right = min(float(width), max(x0, x1))
    bottom = min(float(height), max(y0, y1))
    area = max(0.0, right - left) * max(0.0, bottom - top)
    return float(area / float(height * width))


class GroundingDinoDetector:
    """GroundingDINO based open-vocabulary detector.

    Args:
        config_path: Path to the GroundingDINO model config; falls back to ``GROUNDING_DINO_CONFIG``.
        weights_path: Path to the GroundingDINO checkpoint; falls back to ``GROUNDING_DINO_WEIGHTS``.
        box_threshold: Minimum box confidence kept in the output.
        text_threshold: Minimum token confidence required for phrase matching.
        device: Torch device string; a CUDA request without a visible GPU falls back to CPU.

    Raises:
        RuntimeError: If the config or the checkpoint path cannot be resolved.
        ImportError: If the ``groundingdino`` package is not installed.
    """

    def __init__(
        self,
        config_path: str | Path | None = None,
        weights_path: str | Path | None = None,
        box_threshold: float = 0.3,
        text_threshold: float = 0.25,
        device: str = "cuda",
    ) -> None:
        config_path = config_path or os.environ.get(CONFIG_ENV_VAR)
        weights_path = weights_path or os.environ.get(WEIGHTS_ENV_VAR)
        missing = [
            name
            for name, value in ((CONFIG_ENV_VAR, config_path), (WEIGHTS_ENV_VAR, weights_path))
            if not value
        ]
        if missing:
            raise RuntimeError(
                "GroundingDINO config and weights must be provided either as arguments or through "
                "the " + " and ".join(missing) + " environment variable(s)."
            )
        for name, value in ((CONFIG_ENV_VAR, config_path), (WEIGHTS_ENV_VAR, weights_path)):
            if not Path(str(value)).is_file():
                raise FileNotFoundError(f"{name} points at {value!r}, which is not an existing file")

        self.config_path = str(config_path)
        self.weights_path = str(weights_path)
        self.box_threshold = float(box_threshold)
        self.text_threshold = float(text_threshold)
        self.device = _resolve_device(device)

        transform, load_model, predict = _load_backend()
        self._transform = transform
        self._predict = predict
        self.model = load_model(self.config_path, self.weights_path, device=self.device)
        logger.info(
            "GroundingDINO ready on %s (box_threshold=%.2f, text_threshold=%.2f)",
            self.device,
            self.box_threshold,
            self.text_threshold,
        )

    @classmethod
    def from_env(cls, **overrides: Any) -> "GroundingDinoDetector":
        """Build a detector from the ``GROUNDING_DINO_CONFIG``/``GROUNDING_DINO_WEIGHTS`` variables.

        Args:
            **overrides: Keyword arguments forwarded to :meth:`__init__`; they take precedence over
                the environment.

        Returns:
            A configured detector.

        Raises:
            RuntimeError: If either environment variable is unset.
        """
        missing = [name for name in (CONFIG_ENV_VAR, WEIGHTS_ENV_VAR) if not os.environ.get(name)]
        if missing:
            raise RuntimeError(
                "Missing environment variable(s) "
                + ", ".join(missing)
                + " required to locate the GroundingDINO config and checkpoint."
            )
        overrides.setdefault("config_path", os.environ[CONFIG_ENV_VAR])
        overrides.setdefault("weights_path", os.environ[WEIGHTS_ENV_VAR])
        return cls(**overrides)

    def detect(self, frames: Any, prompt: str) -> list[list[Detection]]:
        """Detect the objects named in ``prompt`` on every frame.

        Args:
            frames: Frames as ``(T, H, W, 3)`` uint8 RGB, a single ``(H, W, 3)`` uint8 RGB image or
                ``(C, T, H, W)`` floats in ``[-1, 1]``.
            prompt: Caption listing the objects to look for.

        Returns:
            One list of :class:`Detection` per frame, sorted by descending score.
        """
        return self._detect_frames(_as_uint8_frames(frames), _as_caption(prompt))

    def detect_batch(self, frames: Any, prompt: str, batch_size: int = 8) -> list[list[Detection]]:
        """Detect ``prompt`` over many frames while bounding peak memory.

        GroundingDINO runs one image per forward pass, so ``batch_size`` only controls how many frames
        are converted per chunk; it is a memory knob rather than a value fixed by the experiments.

        Args:
            frames: Frames as ``(T, H, W, 3)`` uint8 RGB, a single ``(H, W, 3)`` uint8 RGB image or
                ``(C, T, H, W)`` floats in ``[-1, 1]``.
            prompt: Caption listing the objects to look for.
            batch_size: Number of frames converted and processed per chunk.

        Returns:
            One list of :class:`Detection` per frame, in input order.
        """
        array = _as_uint8_frames(frames)
        caption = _as_caption(prompt)
        step = max(1, int(batch_size))
        detections: list[list[Detection]] = []
        for start in range(0, len(array), step):
            detections.extend(self._detect_frames(array[start : start + step], caption))
        return detections

    def _detect_frames(self, frames: np.ndarray, caption: str) -> list[list[Detection]]:
        return [self._detect_frame(frame, caption) for frame in frames]

    def _detect_frame(self, frame: np.ndarray, caption: str) -> list[Detection]:
        from PIL import Image

        height, width = int(frame.shape[0]), int(frame.shape[1])
        image = self._transform(Image.fromarray(frame), None)[0]
        boxes, scores, phrases = self._predict(
            model=self.model,
            image=image,
            caption=caption,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            device=self.device,
        )
        boxes = _numpy(boxes).astype(np.float32).reshape(-1, 4)
        scores = _numpy(scores).astype(np.float32).reshape(-1)
        if boxes.shape[0] == 0:
            return []
        boxes = boxes * np.asarray([width, height, width, height], dtype=np.float32)
        boxes = _clip_boxes(_cxcywh_to_xyxy(boxes), height, width)
        labels = [str(phrase) for phrase in phrases]
        order = np.argsort(-scores, kind="stable")
        return [
            Detection(box=boxes[index], score=float(scores[index]), label=labels[index])
            for index in order
        ]


def _as_caption(prompt: str) -> str:
    """Normalise a prompt into the ``"label. label."`` caption GroundingDINO expects."""
    labels = parse_objects(prompt)
    if not labels:
        raise ValueError("prompt does not contain any object phrase to ground")
    return "".join(label if label.endswith(".") else label + "." for label in labels)


def _load_backend() -> tuple[Any, Any, Any]:
    """Import the GroundingDINO backend and build its input transform.

    Returns:
        ``(transform, load_model, predict)`` taken from the installed package.

    Raises:
        ImportError: If the ``groundingdino`` package is not importable.
    """
    try:
        from groundingdino.datasets import transforms as T
        from groundingdino.util.inference import load_model, predict
    except ImportError as exc:
        raise ImportError(
            "The 'groundingdino' package is required for open-vocabulary detection but is not "
            "installed. Install GroundingDINO from source "
            "(https://github.com/IDEA-Research/GroundingDINO) together with the evaluation extra "
            '(pip install -e ".[eval]"), then point GROUNDING_DINO_CONFIG and GROUNDING_DINO_WEIGHTS '
            "at the config and checkpoint files."
        ) from exc
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return transform, load_model, predict


def _resolve_device(requested: str) -> str:
    """Return ``requested`` unless it asks for CUDA while no GPU is visible.

    Args:
        requested: Device string handed to the detector.

    Returns:
        A device string that torch can address on this machine.
    """
    device = str(requested)
    try:
        import torch
    except ImportError:
        logger.warning("torch is not installed; running GroundingDINO setup for %r", device)
        return device
    if device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("device %r was requested but CUDA is not available; using 'cpu'", device)
        return "cpu"
    return device


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


def _cxcywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """Convert ``(N, 4)`` normalized-centre boxes into top-left/bottom-right form."""
    centres = boxes[:, :2]
    sizes = boxes[:, 2:]
    return np.concatenate([centres - 0.5 * sizes, centres + 0.5 * sizes], axis=1)


def _clip_boxes(boxes: np.ndarray, height: int, width: int) -> np.ndarray:
    """Clip ``(N, 4)`` pixel boxes to the frame and sort each corner pair."""
    lower = np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    upper = np.asarray(
        [max(width - 1, 0), max(height - 1, 0), max(width - 1, 0), max(height - 1, 0)],
        dtype=np.float32,
    )
    clipped = np.clip(boxes, lower, upper)
    x0 = np.minimum(clipped[:, 0], clipped[:, 2])
    y0 = np.minimum(clipped[:, 1], clipped[:, 3])
    x1 = np.maximum(clipped[:, 0], clipped[:, 2])
    y1 = np.maximum(clipped[:, 1], clipped[:, 3])
    return np.stack([x0, y0, x1, y1], axis=1).astype(np.float32)


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
