"""Per-frame fidelity for the RoboTwin cross-backbone study (Table 5).

The three scores follow the FlowWAM metric stack used for the held-out RoboTwin episodes:
PSNR is ``10 * log10(255^2 / mse)`` per frame, averaged over the frames the two clips share;
SSIM averages :data:`DEFAULT_SSIM_SAMPLES` frames spread evenly over the clip; LPIPS uses the
AlexNet backbone on ``256 x 256`` frames rescaled to ``[-1, 1]``, on the same even spread with
:data:`DEFAULT_LPIPS_SAMPLES` frames. Table 5 reports EVEWorld at 12.765 / 0.769 / 0.365
against 12.218 / 0.748 / 0.383 for the FlowWAM Stage-1 control.

Frames are ``(T, H, W, 3)`` uint8 RGB clips or a single ``(H, W, 3)`` frame. SSIM and LPIPS
decode their dependencies lazily, so importing this module costs one numpy import, while
:func:`psnr` and :func:`psnr_clip` need nothing beyond numpy.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from eveworld.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = ["psnr", "psnr_clip", "ssim_clip", "lpips_clip"]

DEFAULT_SSIM_SAMPLES = 16
DEFAULT_LPIPS_SAMPLES = 8
_LPIPS_SIZE = (256, 256)
_EPS = 1e-12

_LPIPS_CACHE: dict[str, Any] = {}


def psnr(pred: np.ndarray, target: np.ndarray, *, data_range: float = 255.0) -> float:
    """Peak signal-to-noise ratio between matching frames.

    Args:
        pred: Predicted frame ``(H, W, 3)`` or clip ``(T, H, W, 3)``; uint8 or float.
        target: Reference array with the same shape as ``pred``.
        data_range: Value range of the samples; 255 for uint8 frames.

    Returns:
        The per-frame PSNR, or the mean of the per-frame values when whole clips are passed.
    """
    pred_frames = _as_frames(pred, "pred")
    target_frames = _as_frames(target, "target")
    if pred_frames.shape != target_frames.shape:
        raise ValueError(f"shape mismatch: {pred_frames.shape} vs {target_frames.shape}")
    if pred_frames.ndim == 3:
        return _frame_psnr(pred_frames, target_frames, data_range)
    values = [_frame_psnr(a, b, data_range) for a, b in zip(pred_frames, target_frames)]
    return float(np.mean(values))


def psnr_clip(pred: np.ndarray, target: np.ndarray, *, data_range: float = 255.0) -> float:
    """Mean per-frame PSNR over the ``min(T_pred, T_target)`` aligned frames of two clips.

    Clips of different length are truncated to the shared prefix, which is how the RoboTwin
    study compares a generated trajectory against the expert trajectory recorded for the task.
    """
    pred_frames = _as_frames(pred, "pred")
    target_frames = _as_frames(target, "target")
    total = min(len(pred_frames), len(target_frames))
    if total == 0:
        raise ValueError("cannot compare empty clips")
    values = [_frame_psnr(pred_frames[i], target_frames[i], data_range) for i in range(total)]
    return float(np.mean(values))


def ssim_clip(
    pred: np.ndarray,
    target: np.ndarray,
    *,
    samples: int = DEFAULT_SSIM_SAMPLES,
    data_range: float = 255.0,
) -> float:
    """Mean structural similarity over evenly spread frames of two clips.

    Args:
        pred: Predicted clip ``(T, H, W, 3)`` uint8 RGB.
        target: Reference clip ``(T, H, W, 3)`` uint8 RGB.
        samples: Number of frames to score; the evenly spread indices are deduplicated, and a
            clip shorter than ``samples`` is scored on every frame.
        data_range: Value range of the samples; 255 for uint8 frames.

    Returns:
        The mean SSIM over the sampled frames.
    """
    from skimage.metrics import structural_similarity

    from eveworld.data.transforms.video import sample_indices

    pred_frames = _as_frames(pred, "pred")
    target_frames = _as_frames(target, "target")
    total = min(len(pred_frames), len(target_frames))
    if total == 0:
        raise ValueError("cannot compare empty clips")
    ids = sample_indices(total, samples)
    values = [
        float(
            structural_similarity(
                pred_frames[i],
                target_frames[i],
                channel_axis=2,
                data_range=data_range,
            )
        )
        for i in ids
    ]
    return float(np.mean(values))


def lpips_clip(
    pred: np.ndarray,
    target: np.ndarray,
    *,
    samples: int = DEFAULT_LPIPS_SAMPLES,
    device: str | None = None,
) -> float:
    """Mean LPIPS (AlexNet) over evenly spread frames of two clips.

    Frames are resized to ``256 x 256`` with bilinear interpolation and rescaled to ``[-1, 1]``
    before the forward pass, matching the FlowWAM evaluation script. The network is built on
    first use and cached per device, since loading it once per clip would dominate the runtime
    of a benchmark sweep.

    Args:
        pred: Predicted clip ``(T, H, W, 3)`` uint8 RGB.
        target: Reference clip ``(T, H, W, 3)`` uint8 RGB.
        samples: Number of frames to score; a clip shorter than ``samples`` is scored on every
            frame.
        device: Torch device string; defaults to CUDA when it is available, else CPU.

    Returns:
        The mean LPIPS distance over the sampled frames; lower is closer.
    """
    import torch
    import torch.nn.functional as functional

    from eveworld.data.transforms.video import sample_indices

    pred_frames = _as_frames(pred, "pred")
    target_frames = _as_frames(target, "target")
    total = min(len(pred_frames), len(target_frames))
    if total == 0:
        raise ValueError("cannot compare empty clips")
    resolved = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = _lpips_model(resolved)
    ids = sample_indices(total, samples)

    def to_tensor(frames: np.ndarray) -> "torch.Tensor":
        batch = np.stack([frames[i] for i in ids])
        tensor = torch.from_numpy(np.ascontiguousarray(batch)).permute(0, 3, 1, 2).float()
        tensor = functional.interpolate(tensor, size=_LPIPS_SIZE, mode="bilinear", align_corners=False)
        return tensor.to(resolved) / 127.5 - 1.0

    with torch.no_grad():
        distance = model(to_tensor(pred_frames), to_tensor(target_frames))
    return float(distance.mean().item())


def _lpips_model(device: str) -> Any:
    """Return the cached AlexNet LPIPS network on ``device``, building it on first use."""
    model = _LPIPS_CACHE.get(device)
    if model is None:
        import lpips

        model = lpips.LPIPS(net="alex").to(device).eval()
        _LPIPS_CACHE[device] = model
        logger.debug("loaded LPIPS (alex) on %s", device)
    return model


def _frame_psnr(pred: np.ndarray, target: np.ndarray, data_range: float) -> float:
    """PSNR of a single frame pair, with the error term floored to avoid a division by zero."""
    diff = pred.astype(np.float64) - target.astype(np.float64)
    mse = float(np.mean(diff * diff))
    return 10.0 * math.log10((data_range * data_range) / max(mse, _EPS))


def _as_frames(array: np.ndarray, name: str) -> np.ndarray:
    """View ``array`` as an ``(T, H, W, 3)`` uint8 array, promoting a single frame to a clip."""
    frames = np.asarray(array)
    if frames.ndim == 3:
        frames = frames[np.newaxis]
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"{name} must be (H, W, 3) or (T, H, W, 3) RGB, got {frames.shape}")
    if frames.dtype != np.uint8:
        raise ValueError(f"{name} must be uint8 RGB, got {frames.dtype}")
    return frames
