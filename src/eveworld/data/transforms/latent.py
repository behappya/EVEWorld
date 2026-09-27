"""Latent-space helpers shared by the IGR sampler, the trainers and the metrics.

The video VAE compresses `(T, H, W, 3)` frames to `(C, T, h, w)` latents with a
spatial stride of 8 and a temporal stride of 4, which for the protocol clips means
`480x768` frames become a `(C, 30, 60, 96)` latent. Two derived views are used:

* the **token grid**, the layout the diffusion transformer actually consumes: the
  latent is folded into `2x2` spatial patches, giving `(h/2 * w/2, T, 4C)`;
* the **weight map**, the per-cell interaction weighting of the IGR objective, which
  lives on the latent grid and is resized to whatever grid a caller needs while
  keeping its unit mean.

The VAE is duck-typed: both the `diffusers` wrapping (`vae.encode(x).latent_dist`) and
a plain `encode`/`decode` pair returning tensors are accepted.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from eveworld.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "decode_latent",
    "encode_latent",
    "latent_grid_size",
    "latent_to_token_grid",
    "resize_weight_map",
    "token_grid_to_latent",
]


def encode_latent(vae: Any, video: Any, sample: bool = False) -> torch.Tensor:
    """Encode frames into a `(C, T, h, w)` latent with the video VAE.

    Args:
        vae: video autoencoder exposing `encode` (the `diffusers` wrapping is
            unwrapped through `latent_dist`).
        video: `(T, H, W, 3)` uint8 RGB frames, or float frames already in `[-1, 1]` with
            either a `(T, C, H, W)` or a `(C, T, H, W)` layout. `numpy` and `torch`
            inputs are both accepted.
        sample: draw from the posterior instead of using its mode, which is what the
            stochastic latent caches use.

    Returns:
        `(C, T, h, w)` float32 latent of the single clip.
    """
    tensor = _as_video_tensor(video)
    with torch.no_grad():
        latent = _unwrap_latent(_encode(vae, tensor), sample=sample)
    if latent.ndim == 5:
        latent = latent[0]
    if latent.ndim != 4:
        raise ValueError(f"Expected a (C, T, h, w) latent, got {tuple(latent.shape)}")
    return latent


def decode_latent(vae: Any, latent: Any) -> torch.Tensor:
    """Decode a latent back to frames with the video VAE.

    Args:
        vae: video autoencoder exposing `decode`.
        latent: `(C, T, h, w)` latent of a single clip, or `(B, C, T, h, w)` for a batch.

    Returns:
        Frames in the layout the VAE returns: `(C, T, H, W)` values in `[-1, 1]` for a
        4-D input, and the batched counterpart for a 5-D input.
    """
    tensor = latent if torch.is_tensor(latent) else torch.as_tensor(latent)
    unbatched = tensor.ndim == 4
    if unbatched:
        tensor = tensor[None]
    elif tensor.ndim != 5:
        raise ValueError(f"Expected (C, T, h, w) or (B, C, T, h, w), got {tuple(tensor.shape)}")
    with torch.no_grad():
        frames = _unwrap_latent(_decode(vae, tensor), sample=False)
    if unbatched and frames.ndim == 5:
        frames = frames[0]
    return frames


def latent_grid_size(size: tuple[int, int], stride: int = 8) -> tuple[int, int]:
    """Size of the latent grid for a frame of `size` pixels.

    Args:
        size: `(height, width)` in pixels, the order used for the protocol resolutions
            (`(480, 768)`, `(480, 640)`).
        stride: spatial compression factor of the VAE, 8 for the video VAE.

    Returns:
        `(height // stride, width // stride)`, e.g. `(60, 96)` for `(480, 768)`. A size
        that is not a multiple of the stride is floored, which is what the patchified
        VAE does internally.
    """
    stride = int(stride)
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")
    height, width = int(size[0]), int(size[1])
    return height // stride, width // stride


def latent_to_token_grid(latent: torch.Tensor) -> torch.Tensor:
    """Fold a `(C, T, h, w)` latent into the `(h/2 * w/2, T, 4C)` token grid.

    The spatial grid is patchified with the same `2x2` patch order the diffusion
    transformer uses, and the channel dimension is folded into the token embedding,
    so token `(i, j)` carries the four latents of its patch.

    Args:
        latent: `(C, T, h, w)` latent, with `h` and `w` even.

    Returns:
        `(h // 2 * w // 2, T, 4 * C)` tensor.
    """
    tensor = latent if torch.is_tensor(latent) else torch.as_tensor(latent)
    if tensor.ndim != 4:
        raise ValueError(f"Expected (C, T, h, w), got {tuple(tensor.shape)}")
    channels, frames, height, width = tensor.shape
    if height % 2 or width % 2:
        raise ValueError(f"Latent height and width must be even, got {(height, width)}")
    half_h, half_w = height // 2, width // 2
    grid = tensor.reshape(channels, frames, half_h, 2, half_w, 2)
    return grid.permute(2, 4, 1, 3, 5, 0).reshape(half_h * half_w, frames, 4 * channels)


def token_grid_to_latent(tokens: torch.Tensor, shape: tuple[int, int, int, int]) -> torch.Tensor:
    """Undo :func:`latent_to_token_grid`.

    Args:
        tokens: `(h // 2 * w // 2, T, 4 * C)` token grid.
        shape: `(C, T, h, w)` of the latent to rebuild, with even `h` and `w`.

    Returns:
        `(C, T, h, w)` latent tensor.
    """
    tensor = tokens if torch.is_tensor(tokens) else torch.as_tensor(tokens)
    channels, frames, height, width = (int(v) for v in shape)
    if height % 2 or width % 2:
        raise ValueError(f"Latent height and width must be even, got {(height, width)}")
    half_h, half_w = height // 2, width // 2
    expected = (half_h * half_w, frames, 4 * channels)
    if tuple(tensor.shape) != expected:
        raise ValueError(f"Expected a {expected} token grid, got {tuple(tensor.shape)}")
    grid = tensor.reshape(half_h, half_w, frames, 2, 2, channels)
    return grid.permute(5, 2, 0, 3, 1, 4).reshape(channels, frames, height, width)


def resize_weight_map(weight_map: Any, size: tuple[int, int]) -> Any:
    """Resize a weight map to `size` while keeping its unit mean.

    The map is sampled with nearest neighbours, which reproduces the `2x2` block
    upsampling the IGR implementation applies to the latent grid exactly, and keeps the
    discrete levels of the map (the paper's background `1.0` and disturbed `3.0`)
    instead of blending them. The resized map is divided by its own mean, so the mean
    loss scale is unchanged by the resolution.

    Args:
        weight_map: `(H, W)` or `(T, H, W)` map, `numpy` or `torch`. Higher-rank layouts
            are resized on their last two dimensions. `numpy` in returns `numpy` out,
            `torch` in returns `torch` out.
        size: `(height, width)` of the output grid.

    Returns:
        The resized map, unit mean, in the layout and library of the input.
    """
    height, width = int(size[0]), int(size[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"size must be positive, got {size}")
    tensor_in = torch.is_tensor(weight_map)
    array = weight_map if tensor_in else np.asarray(weight_map, dtype=np.float32)
    if array.ndim < 2:
        raise ValueError(f"Expected at least a (H, W) map, got {tuple(array.shape)}")
    source_h, source_w = int(array.shape[-2]), int(array.shape[-1])
    if (source_h, source_w) == (height, width):
        resized = array
    else:
        resized = _nearest_resize(array, height, width, tensor_in)
    if tensor_in:
        return resized / resized.mean().clamp_min(1e-6)
    mean = float(np.clip(np.mean(resized), 1e-6, None))
    return (resized / mean).astype(np.float32)


def _nearest_resize(array: Any, height: int, width: int, tensor_in: bool) -> Any:
    """Nearest-neighbour resize of the last two dimensions of `array`."""
    rows = _source_indices(int(array.shape[-2]), height)
    cols = _source_indices(int(array.shape[-1]), width)
    if tensor_in:
        rows = torch.from_numpy(rows)
        cols = torch.from_numpy(cols)
    return array[..., rows, :][..., :, cols]


def _source_indices(source: int, target: int) -> np.ndarray:
    """Source index of every target cell, nearest neighbours, integer factors exact."""
    positions = (np.arange(target, dtype=np.float64) + 0.5) * (source / target)
    return np.clip(positions.astype(np.int64), 0, max(source - 1, 0))


def _encode(vae: Any, video: torch.Tensor) -> Any:
    """Call the encoder of `vae` on a `(1, C, T, H, W)` tensor."""
    encoder = getattr(vae, "encode", None) or getattr(vae, "encode_", None)
    if encoder is None:
        raise AttributeError("The VAE exposes neither encode() nor encode_()")
    try:
        return encoder(video, return_dict=False)
    except TypeError:
        return encoder(video)


def _decode(vae: Any, latent: torch.Tensor) -> Any:
    """Call the decoder of `vae` on a `(1, C, T, h, w)` tensor."""
    decoder = getattr(vae, "decode", None) or getattr(vae, "decode_", None)
    if decoder is None:
        raise AttributeError("The VAE exposes neither decode() nor decode_()")
    try:
        return decoder(latent, return_dict=False)
    except TypeError:
        return decoder(latent)


def _unwrap_latent(output: Any, sample: bool) -> torch.Tensor:
    """Extract a latent tensor from a VAE output, whatever wrapper it arrives in.

    Handles a bare tensor, a `(latent,)` tuple, the `diffusers` `latent_dist`, and the
    `DecoderOutput` whose `sample` attribute is itself a tensor.
    """
    if isinstance(output, (tuple, list)):
        if not output:
            raise ValueError("The VAE returned an empty output")
        output = output[0]
    distribution = getattr(output, "latent_dist", None)
    if distribution is not None:
        output = distribution
    sampler = getattr(output, "sample", None)
    if sampler is not None and callable(sampler):
        if sample:
            return sampler()
        for name in ("mode", "mean"):
            value = getattr(output, name, None)
            if value is None:
                continue
            return value() if callable(value) else value
        return sampler()
    if torch.is_tensor(output):
        return output
    value = getattr(output, "sample", None)
    if torch.is_tensor(value):
        return value
    if hasattr(output, "to") and hasattr(output, "shape"):
        return output
    raise TypeError(f"Cannot read a latent tensor out of {type(output).__name__}")


def _as_video_tensor(video: Any) -> torch.Tensor:
    """Turn frames into a `(1, C, T, H, W)` float32 tensor in `[-1, 1]`."""
    if torch.is_tensor(video):
        tensor = video.detach().to(device="cpu", dtype=torch.float32)
    else:
        tensor = torch.from_numpy(np.asarray(video, dtype=np.float32))
    if tensor.ndim == 4 and tensor.shape[-1] == 3:
        tensor = tensor.permute(3, 0, 1, 2)
    elif tensor.ndim == 4 and tensor.shape[1] == 3:
        tensor = tensor.permute(1, 0, 2, 3)
    elif tensor.ndim != 4 or tensor.shape[0] != 3:
        raise ValueError(f"Expected (T, H, W, 3) frames or a (C, T, H, W) tensor, got {tuple(tensor.shape)}")
    return _to_unit_range(tensor)[None]


def _to_unit_range(tensor: torch.Tensor) -> torch.Tensor:
    """Map frames to `[-1, 1]`, the range the video VAE was trained on."""
    if float(tensor.min()) >= 0.0 and float(tensor.max()) > 1.5:
        return tensor / 127.5 - 1.0
    if float(tensor.min()) >= 0.0:
        return tensor * 2.0 - 1.0
    return tensor
