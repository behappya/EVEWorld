"""Video decoding, encoding and frame sampling.

Frames are `np.ndarray` in `(T, H, W, 3)` uint8 RGB, the convention shared by the
benchmarks, the IGR sampler and the trainers. Decoding goes through PyAV, with an
`imageio` fallback for environments where PyAV is not installed.

`sample_indices` reproduces the frame-index rule of the WorldArena protocol: the
`"round"` mode is the paper protocol and `"linspace"` keeps the truncated
`numpy.linspace` behaviour used by the latent extraction scripts.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import numpy as np
import torch

from eveworld.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "SAMPLE_MODES",
    "frames_to_tensor",
    "load_video",
    "resize_frames",
    "sample_indices",
    "save_video",
    "tensor_to_frames",
]

SAMPLE_MODES = ("linspace", "round", "uniform")


def load_video(
    path: str | Path,
    start: int | None = None,
    length: int | None = None,
    size: tuple[int, int] | None = None,
    out_fps: float | None = None,
) -> np.ndarray:
    """Decode a video file into `(T, H, W, 3)` uint8 RGB frames.

    Args:
        path: video file readable by PyAV (mp4, mkv, ...) or by imageio.
        start: index of the first frame to keep, counted from the beginning.
        length: number of frames to keep from `start` onwards.
        size: `(width, height)` the frames are resized to.
        out_fps: resample the clip to this frame rate by nearest source frame.

    Returns:
        `(T, H, W, 3)` uint8 RGB frames. `T` is 0 only if the file holds no frames,
        in which case a `ValueError` is raised instead.
    """
    source = Path(path)
    if length is not None and int(length) <= 0:
        raise ValueError(f"length must be positive, got {length}")
    frames, source_fps = _read_frames(source, start, length)
    if frames.shape[0] == 0:
        raise ValueError(f"No decodable frames in {source}")
    if size is not None:
        frames = resize_frames(frames, size)
    if out_fps is not None and frames.shape[0] > 1:
        frames = _resample_frames(frames, source_fps, float(out_fps))
    return frames


def save_video(frames: np.ndarray, path: str | Path, fps: float = 16.0, crf: int = 18) -> Path:
    """Write `(T, H, W, 3)` uint8 RGB frames to `path` as H.264.

    Frames are cropped to even height and width, which `yuv420p` requires; `crf`
    is the libx264 constant-rate-factor quality knob.

    Args:
        frames: `(T, H, W, 3)` uint8 RGB frames.
        path: output file; parent directories are created when missing.
        fps: playback frame rate written into the container.
        crf: libx264 quality, lower is better.

    Returns:
        The output path.
    """
    array = np.asarray(frames)
    if array.ndim != 4 or array.shape[-1] != 3:
        raise ValueError(f"Expected (T, H, W, 3) frames, got {array.shape}")
    height = array.shape[1] - array.shape[1] % 2
    width = array.shape[2] - array.shape[2] % 2
    if height == 0 or width == 0:
        raise ValueError(f"Frames must be at least 2x2 pixels, got {array.shape[1:3]}")
    array = np.ascontiguousarray(array[:, :height, :width].astype(np.uint8, copy=False))
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        _write_with_av(array, target, fps, crf)
    except ImportError:
        logger.debug("PyAV is unavailable; encoding %s with imageio", target)
        _write_with_imageio(array, target, fps)
    return target


def frames_to_tensor(frames: np.ndarray, normalize: bool = True) -> torch.Tensor:
    """Convert `(T, H, W, 3)` uint8 RGB frames to a `(C, T, H, W)` float32 tensor.

    Args:
        frames: `(T, H, W, 3)` uint8 RGB frames.
        normalize: map `[0, 255]` to `[-1, 1]` instead of `[0, 1]`, which is what
            the VAE and the diffusion backbone expect.
    """
    array = np.asarray(frames, dtype=np.float32) / 255.0
    if normalize:
        array = array * 2.0 - 1.0
    return torch.from_numpy(np.ascontiguousarray(array.transpose(3, 0, 1, 2)))


def tensor_to_frames(tensor: torch.Tensor) -> np.ndarray:
    """Convert a `(C, T, H, W)` tensor back to `(T, H, W, 3)` uint8 RGB frames.

    Values are read as `[-1, 1]` when the tensor is signed, otherwise as `[0, 1]`.
    """
    if torch.is_tensor(tensor):
        array = tensor.detach().to(device="cpu", dtype=torch.float32).numpy()
    else:
        array = np.asarray(tensor, dtype=np.float32)
    if array.ndim != 4:
        raise ValueError(f"Expected (C, T, H, W), got {array.shape}")
    if array.shape[0] != 3:
        if array.shape[1] == 3:
            array = array.transpose(1, 0, 2, 3)
        else:
            raise ValueError(f"Expected (C, T, H, W) with C == 3, got {array.shape}")
    if float(array.min()) < 0.0:
        array = (array + 1.0) * 0.5
    array = np.clip(array, 0.0, 1.0)
    return np.round(array * 255.0).astype(np.uint8).transpose(1, 2, 3, 0)


def resize_frames(frames: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize `(T, H, W, 3)` frames, or a single `(H, W, 3)` frame, to `size`.

    `size` is `(width, height)`, matching `cv2.resize` and `PIL.Image.resize`,
    which is the order used by the extraction scripts.
    """
    array = np.asarray(frames)
    single = array.ndim == 3
    if single:
        array = array[None]
    if array.ndim != 4 or array.shape[-1] != 3:
        raise ValueError(f"Expected (T, H, W, 3) frames, got {array.shape}")
    width, height = int(size[0]), int(size[1])
    if width <= 0 or height <= 0:
        raise ValueError(f"size must be positive, got {size}")
    if (height, width) == array.shape[1:3]:
        resized = np.array(array)
    else:
        resized = _resize_stack(array, width, height)
    return resized[0] if single else resized


def sample_indices(num_frames: int, num_out: int, mode: str = "linspace") -> np.ndarray:
    """Source-frame indices used to build a fixed-length clip.

    The `"round"` mode rounds `i * (F - 1) / (n - 1)` to the nearest frame (the
    WorldArena protocol); `"linspace"` keeps the truncated `numpy.linspace`
    behaviour, and `"uniform"` is an alias for it.

    Args:
        num_frames: number of frames in the source clip.
        num_out: number of frames wanted.
        mode: one of `SAMPLE_MODES`.

    Returns:
        `(num_out,)` int64 indices in ascending order, clipped to `[0, num_frames - 1]`.
    """
    if mode not in SAMPLE_MODES:
        raise ValueError(f"Unknown sample mode: {mode!r}; expected one of {SAMPLE_MODES}")
    num_frames = int(num_frames)
    num_out = int(num_out)
    if num_frames <= 0:
        return np.zeros(0, dtype=np.int64)
    if num_out <= 1:
        return np.array([0], dtype=np.int64)
    last = num_frames - 1
    if mode in ("linspace", "uniform"):
        indices = np.linspace(0, last, num_out).astype(np.int64)
    else:
        steps = np.arange(num_out, dtype=np.float64) * last / (num_out - 1)
        indices = np.floor(steps + 0.5).astype(np.int64)
    return np.clip(indices, 0, last).astype(np.int64)


def _read_frames(
    path: Path, start: int | None, length: int | None
) -> tuple[np.ndarray, float]:
    """Decode frames with PyAV, falling back to imageio when PyAV is missing."""
    try:
        return _read_with_av(path, start, length)
    except ImportError:
        logger.debug("PyAV is unavailable; decoding %s with imageio", path)
        return _read_with_imageio(path, start, length)


def _read_with_av(path: Path, start: int | None, length: int | None) -> tuple[np.ndarray, float]:
    import av

    frames: list[np.ndarray] = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate else 0.0
        for index, frame in enumerate(container.decode(stream)):
            if start is not None and index < int(start):
                continue
            if length is not None and len(frames) >= int(length):
                break
            frames.append(frame.to_ndarray(format="rgb24"))
    return _stack_frames(frames), fps


def _read_with_imageio(
    path: Path, start: int | None, length: int | None
) -> tuple[np.ndarray, float]:
    from imageio.v3 import immeta, imiter

    metadata = immeta(str(path))
    fps = float(metadata.get("fps") or 0.0)
    frames: list[np.ndarray] = []
    for index, frame in enumerate(imiter(str(path))):
        if start is not None and index < int(start):
            continue
        if length is not None and len(frames) >= int(length):
            break
        frames.append(np.asarray(frame)[..., :3].astype(np.uint8))
    return _stack_frames(frames), fps


def _write_with_av(frames: np.ndarray, path: Path, fps: float, crf: int) -> None:
    import av

    rate = Fraction(float(fps)).limit_denominator(1000)
    if float(rate) <= 0.0:
        raise ValueError(f"fps must be positive, got {fps}")
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=rate)
        stream.width = int(frames.shape[2])
        stream.height = int(frames.shape[1])
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": str(int(crf))}
        for frame in frames:
            packet = av.VideoFrame.from_ndarray(np.ascontiguousarray(frame), format="rgb24")
            for encoded in stream.encode(packet):
                container.mux(encoded)
        for encoded in stream.encode():
            container.mux(encoded)


def _write_with_imageio(frames: np.ndarray, path: Path, fps: float) -> None:
    from imageio.v3 import imwrite

    imwrite(str(path), np.asarray(frames), fps=float(fps), codec="libx264")


def _resample_frames(frames: np.ndarray, source_fps: float, out_fps: float) -> np.ndarray:
    """Nearest-source-frame resampling of `(T, H, W, 3)` frames."""
    if out_fps <= 0.0:
        raise ValueError(f"out_fps must be positive, got {out_fps}")
    if source_fps <= 0.0:
        raise ValueError("Cannot resample a video with an unknown frame rate")
    count = max(1, int(round(frames.shape[0] * out_fps / source_fps)))
    positions = np.floor(np.arange(count, dtype=np.float64) * source_fps / out_fps)
    indices = np.clip(positions.astype(np.int64), 0, frames.shape[0] - 1)
    return frames[indices]


def _resize_stack(array: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize a `(T, H, W, 3)` stack with OpenCV, falling back to Pillow."""
    downscaling = width * height < array.shape[1] * array.shape[2]
    try:
        import cv2

        interpolation = cv2.INTER_AREA if downscaling else cv2.INTER_LINEAR
        return np.stack(
            [cv2.resize(frame, (width, height), interpolation=interpolation) for frame in array]
        )
    except ImportError:
        from PIL import Image

        resample = Image.BILINEAR
        return np.stack(
            [np.asarray(Image.fromarray(frame).resize((width, height), resample)) for frame in array]
        )


def _stack_frames(frames: list[np.ndarray]) -> np.ndarray:
    if not frames:
        return np.zeros((0, 0, 0, 3), dtype=np.uint8)
    return np.stack([np.asarray(frame, dtype=np.uint8) for frame in frames])
