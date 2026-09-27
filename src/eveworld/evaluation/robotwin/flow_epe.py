"""Optical-flow end-point error for the RoboTwin transfer study (Table 5).

Flow is the strongest signal in the table: EVEWorld reaches 2.207 px against 3.033 px for the
FlowWAM Stage-1 control, because a world model that predicts temporally consistent motion needs
less motion supervision at inference time. Both clips of a pair are converted to dense RAFT flow
and compared pixel by pixel with the end-point error ``||flow_pred - flow_target||_2``; the score
of a pair is the mean error over the image, and the score of a clip is the mean over its
consecutive frame pairs.

Flow tensors are ``(2, H, W)`` per pair or ``(T - 1, 2, H, W)`` for a clip of ``T`` frames, with
channel 0 the horizontal and channel 1 the vertical displacement in pixels. The ``(H, W, 2)``
layout that RAFT and OpenCV use is accepted everywhere flow is an input, so a map produced by an
external extractor can be scored without a conversion step. RAFT itself lives in the FlowWAM
checkout and is imported on first use, which keeps the module importable with numpy alone.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from eveworld.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = ["compute_flow", "epe_map", "flow_epe"]

DEFAULT_RAFT_BATCH_SIZE = 16


def compute_flow(
    frames: np.ndarray,
    *,
    raft: Any = None,
    max_batch_size: int = DEFAULT_RAFT_BATCH_SIZE,
    device: str | None = None,
) -> np.ndarray:
    """Dense optical flow between every consecutive pair of ``frames``.

    Args:
        frames: Clip ``(T, H, W, 3)`` uint8 RGB, with ``T >= 2``.
        raft: Flow extractor to use. Defaults to
            :class:`~raft_flow_extractor.RAFTFlowExtractor` from the FlowWAM checkout; an object
            exposing ``batch_call(frames, max_batch_size=...)`` or a plain
            ``callable(frame_a, frame_b)`` both work, which lets tests score synthetic flow
            without loading the network.
        max_batch_size: Frame pairs per RAFT forward pass.
        device: Torch device for the default extractor; defaults to CUDA when it is available,
            else CPU.

    Returns:
        ``(T - 1, 2, H, W)`` float32 flow, where entry ``i`` maps frame ``i`` to frame ``i + 1``.

    Raises:
        ValueError: If ``frames`` is not a uint8 RGB clip of at least two frames.
        RuntimeError: If the default extractor is needed but the FlowWAM checkout is missing.
    """
    clip = _as_clip(frames)
    extractor = raft if raft is not None else _load_raft(device)
    if hasattr(extractor, "batch_call"):
        flows = list(extractor.batch_call(list(clip), max_batch_size=max_batch_size))
    else:
        flows = [extractor(clip[i], clip[i + 1]) for i in range(len(clip) - 1)]
    expected = len(clip) - 1
    if len(flows) != expected:
        raise RuntimeError(f"extractor returned {len(flows)} flow fields for {expected} pairs")
    stacked = np.stack([_as_channel_first(flow, f"flow[{i}]") for i, flow in enumerate(flows)])
    logger.debug("computed flow for %d pairs of shape %s", expected, stacked.shape[1:])
    return np.ascontiguousarray(stacked, dtype=np.float32)


def epe_map(pred_flow: np.ndarray, target_flow: np.ndarray) -> np.ndarray:
    """End-point error map of one flow pair.

    Args:
        pred_flow: Predicted flow ``(2, H, W)`` or ``(H, W, 2)``.
        target_flow: Reference flow in either layout, with the same shape as ``pred_flow``.

    Returns:
        ``(H, W)`` float32 map holding ``||pred - target||_2`` at every pixel.
    """
    pred = _as_channel_first(pred_flow, "pred_flow")
    target = _as_channel_first(target_flow, "target_flow")
    if pred.shape != target.shape:
        raise ValueError(f"flow shape mismatch: {pred.shape} vs {target.shape}")
    return np.linalg.norm(pred - target, axis=0).astype(np.float32)


def flow_epe(
    pred_flow: np.ndarray,
    target_flow: np.ndarray,
    valid: np.ndarray | None = None,
) -> float:
    """Mean end-point error over a flow pair or a whole clip.

    Args:
        pred_flow: Predicted flow ``(2, H, W)``, ``(H, W, 2)`` or ``(T - 1, 2, H, W)``.
        target_flow: Reference flow with the same shape as ``pred_flow``.
        valid: Boolean mask of the pixels to score. Either ``(H, W)``, broadcast over the
            frames of a clip, or the full flow-frame shape ``(T - 1, H, W)``. When omitted every
            pixel contributes, matching the FlowWAM evaluation script.

    Returns:
        The mean end-point error in pixels, or ``0.0`` when the mask selects no pixel.
    """
    pred = _as_channel_first(pred_flow, "pred_flow")
    target = _as_channel_first(target_flow, "target_flow")
    if pred.shape != target.shape:
        raise ValueError(f"flow shape mismatch: {pred.shape} vs {target.shape}")
    errors = np.linalg.norm(pred - target, axis=-3)
    if valid is None:
        return float(errors.mean()) if errors.size else 0.0
    mask = np.asarray(valid, dtype=bool)
    if mask.shape != errors.shape:
        if errors.ndim == 3 and mask.shape == errors.shape[1:]:
            mask = np.broadcast_to(mask, errors.shape)
        else:
            raise ValueError(f"valid mask {mask.shape} does not match flow frames {errors.shape}")
    if not mask.any():
        return 0.0
    return float(errors[mask].mean())


def _load_raft(device: str | None = None) -> Any:
    """Build the FlowWAM RAFT extractor, naming the checkout when it is unavailable."""
    try:
        from raft_flow_extractor import RAFTFlowExtractor
    except ImportError as exc:
        raise RuntimeError(
            "RAFT flow extraction needs the FlowWAM checkout: run "
            "scripts/setup/clone_flowwam.sh and put third_party/FlowWAM/inference on "
            "PYTHONPATH (or reinstall the package with the 'eval' extra)"
        ) from exc
    if device is None:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    return RAFTFlowExtractor(device=device)


def _as_channel_first(array: np.ndarray, name: str) -> np.ndarray:
    """Return ``array`` as ``(2, H, W)`` or ``(T - 1, 2, H, W)`` float32 flow."""
    flow = np.asarray(array, dtype=np.float32)
    if flow.ndim == 3:
        if flow.shape[0] != 2 and flow.shape[-1] == 2:
            flow = flow.transpose(2, 0, 1)
        if flow.shape[0] != 2:
            raise ValueError(f"{name} must be (2, H, W) or (H, W, 2), got {flow.shape}")
    elif flow.ndim == 4:
        if flow.shape[1] != 2 and flow.shape[-1] == 2:
            flow = flow.transpose(0, 3, 1, 2)
        if flow.shape[1] != 2:
            raise ValueError(f"{name} must be (T, 2, H, W) or (T, H, W, 2), got {flow.shape}")
    else:
        raise ValueError(f"{name} must have 3 or 4 dimensions, got {flow.shape}")
    return np.ascontiguousarray(flow)


def _as_clip(frames: np.ndarray) -> np.ndarray:
    """Validate an ``(T, H, W, 3)`` uint8 clip with at least two frames."""
    clip = np.asarray(frames)
    if clip.ndim != 4 or clip.shape[-1] != 3:
        raise ValueError(f"frames must be (T, H, W, 3) RGB, got {clip.shape}")
    if clip.dtype != np.uint8:
        raise ValueError(f"frames must be uint8 RGB, got {clip.dtype}")
    if len(clip) < 2:
        raise ValueError(f"need at least two frames to compute flow, got {len(clip)}")
    return clip
