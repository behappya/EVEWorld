"""GigaWorld-0 as the EVEWorld video backbone.

Three things live here: the geometry of the backbone, the :class:`GigaWorldConfig` that
mirrors the `model` block of `configs/paper/gigaworld/**`, and :class:`GigaWorldModel`, the
wrapper the trainer talks to.

Geometry. The backbone consumes 16 px spatial cells and covers four frames per temporal
cell, so a 93-frame `480x768` clip yields the `(24, 30, 48)` cell grid the IGR annotations
are written on, while its video VAE produces a `(16, 24, 60, 96)` latent at stride 8. The
IGR weight map is built on the cell grid and resized by the restoration loss to the latent
grid, so the two strides stay apart: :func:`latent_grid` returns the cell grid and
:func:`latent_shape` the VAE latent.

Denoising. `docs/training.md` trains with the EDM objective: additive noise
`x = z + sigma * eps` on the clean latent, the preconditioning
`c_in = 1/sqrt(sigma^2 + sigma_data^2)`, `c_skip = sigma_data^2/(sigma^2 + sigma_data^2)`,
`c_out = sigma*sigma_data/sqrt(sigma^2 + sigma_data^2)` and `c_noise = ln(sigma)/4`, with
the EDM weight `lambda(sigma)` applied by :func:`eveworld.methods.igr.loss.igr_loss`. The
classifier-free combination happens on the raw network output, before `c_skip` and `c_out`,
which equals combining the preconditioned estimates because neither coefficient depends on
the conditioning.

The wrapper is duck-typed on purpose: the released pipeline exposes its denoiser, VAE and
text encoder under names that differ between releases, so the lookups try the known
spellings and raise a `RuntimeError` listing what the object does carry instead of failing
with an `AttributeError` deep inside a training step.
"""

from __future__ import annotations

import inspect
import math
import os
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from eveworld.utils.config import cfg_get
from eveworld.utils.io import repo_root
from eveworld.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "CELL_SIZE",
    "DEFAULT_MODEL_NAME",
    "EDM_P_MEAN",
    "EDM_P_STD",
    "LATENT_CHANNELS",
    "SIGMA_DATA",
    "TEMPORAL_STRIDE",
    "VAE_SPATIAL_STRIDE",
    "GigaWorldConfig",
    "GigaWorldModel",
    "latent_frames",
    "latent_grid",
    "latent_shape",
    "load_backbone",
]


CELL_SIZE = 16
"""Side of a cell of the backbone token grid, in pixels."""

TEMPORAL_STRIDE = 4
"""Frames covered by one temporal cell of the backbone token grid."""

VAE_SPATIAL_STRIDE = 8
"""Spatial compression of the video VAE applied to the pixel frames."""

LATENT_CHANNELS = 16
"""Channels of the video latent the VAE produces."""

SIGMA_DATA = 0.5
"""Standard deviation of the training data, the `sigma_data` of the EDM objective."""

EDM_P_MEAN = -1.2
"""`p_mean` of the log-normal noise-level distribution, the EDM default of the protocol."""

EDM_P_STD = 1.2
"""`p_std` of the log-normal noise-level distribution, the EDM default of the protocol."""

DEFAULT_MODEL_NAME = "Video-Pretrain-2B"
"""Released GigaWorld-0 checkpoint named by `model.checkpoint` in every configuration."""

_BACKBONE_MODULE = "giga_models"
_BACKBONE_CLASS = "GigaWorld0Pipeline"
_THIRD_PARTY_CHECKOUT = Path("third_party") / "giga-world-0"

_VARIANT_PRESETS: dict[str, dict[str, Any]] = {
    "dreamgen": {
        "checkpoint": DEFAULT_MODEL_NAME,
        "resolution": (768, 480),
        "num_frames": 93,
        "fps": 16.0,
        "block_index": 23,
        "max_steps": 250,
    },
    "agibot": {
        "checkpoint": DEFAULT_MODEL_NAME,
        "resolution": (640, 480),
        "num_frames": 93,
        "fps": 16.0,
        "block_index": 23,
        "max_steps": 50,
    },
}


@dataclass
class GigaWorldConfig:
    """Geometry and checkpoint of a GigaWorld-0 run.

    Fields left at `None` are filled from the preset of `variant`, so
    `GigaWorldConfig(variant="agibot")` carries the `480x640` geometry of the transfer
    experiment and `GigaWorldConfig()` the `480x768` geometry of DreamGen.

    Attributes:
        variant: `"dreamgen"` or `"agibot"`; selects which preset fills the unset fields.
        backbone: Name of the backbone family, `"gigaworld0"` in every released config.
        checkpoint: Directory name of the released weights, resolved by
            :func:`load_backbone` under `GW0_MODEL_DIR`, `EVEWORLD_CHECKPOINT_ROOT` or
            `checkpoints/`.
        vae_path: Optional directory holding a video VAE that overrides the one the
            checkpoint ships with.
        text_encoder_path: Optional directory holding a text encoder that overrides the one
            the checkpoint ships with.
        resolution: `(width, height)` of the frames, in pixels, both a multiple of
            :data:`CELL_SIZE`.
        num_frames: Frames of a training clip, 93 for both released variants.
        fps: Frame rate the backbone is conditioned on.
        block_index: Transformer block TIA is attached to, 23 for both released variants.
        max_steps: Optimisation steps of the released run, 250 on DreamGen and 50 on
            the AgiBot transfer set.
        num_steps: Sampling steps of the released generation protocol.
        cfg_scale: Classifier-free guidance scale of the released generation protocol.

    Raises:
        ValueError: If the variant is unknown, the resolution is not a positive multiple of
            the cell size, or a frame count, frame rate or block index is out of range.
    """

    variant: str = "dreamgen"
    backbone: str = "gigaworld0"
    checkpoint: str | None = None
    vae_path: str | None = None
    text_encoder_path: str | None = None
    resolution: tuple[int, int] | None = None
    num_frames: int | None = None
    fps: float | None = None
    block_index: int | None = None
    max_steps: int | None = None
    num_steps: int = 30
    cfg_scale: float = 7.0

    def __post_init__(self) -> None:
        self.variant = str(self.variant).strip().lower()
        if self.variant not in _VARIANT_PRESETS:
            raise ValueError(
                f"Unknown GigaWorld-0 variant {self.variant!r}; expected one of "
                f"{sorted(_VARIANT_PRESETS)}"
            )
        for name, value in _VARIANT_PRESETS[self.variant].items():
            if getattr(self, name) is None:
                setattr(self, name, value)
        self.backbone = str(self.backbone)
        self.checkpoint = str(DEFAULT_MODEL_NAME if self.checkpoint is None else self.checkpoint)
        self.vae_path = None if self.vae_path is None else str(self.vae_path)
        self.text_encoder_path = None if self.text_encoder_path is None else str(self.text_encoder_path)
        self.resolution = _as_resolution(self.resolution)
        self.num_frames = int(self.num_frames)
        self.fps = float(self.fps)
        self.block_index = int(self.block_index)
        self.max_steps = int(self.max_steps)
        self.num_steps = int(self.num_steps)
        self.cfg_scale = float(self.cfg_scale)
        if self.num_frames < 2:
            raise ValueError(f"num_frames must be at least 2, got {self.num_frames}")
        if not math.isfinite(self.fps) or self.fps <= 0.0:
            raise ValueError(f"fps must be a positive rate, got {self.fps}")
        if self.block_index < 0:
            raise ValueError(f"block_index must be non-negative, got {self.block_index}")
        if self.max_steps < 1:
            raise ValueError(f"max_steps must be positive, got {self.max_steps}")
        if self.num_steps < 1:
            raise ValueError(f"num_steps must be positive, got {self.num_steps}")
        if not math.isfinite(self.cfg_scale) or self.cfg_scale < 0.0:
            raise ValueError(f"cfg_scale must be a non-negative scale, got {self.cfg_scale}")

    @classmethod
    def from_config(cls, cfg: Any, **overrides: Any) -> GigaWorldConfig:
        """Read the configuration out of a parsed run config.

        Args:
            cfg: Parsed YAML of a run, e.g. `configs/paper/gigaworld/dreamgen/eveworld.yaml`,
                as a mapping or a `DictConfig`. The `model` block carries the geometry, the
                `train` block the step budget and the `inference` block the sampling
                protocol of the released checkpoint.
            **overrides: Fields that take precedence over the parsed values.

        Returns:
            The configuration; keys the run does not carry keep the preset or dataclass
            default, and the variant is inferred from the run name, the split file and the
            metadata directory when `model.variant` is absent.
        """
        values: dict[str, Any] = {
            "variant": _infer_variant(cfg),
            "backbone": cfg_get(cfg, "model.backbone", "gigaworld0"),
            "checkpoint": cfg_get(cfg, "model.checkpoint"),
            "vae_path": cfg_get(cfg, "model.vae_path"),
            "text_encoder_path": cfg_get(cfg, "model.text_encoder_path"),
            "resolution": cfg_get(cfg, "model.resolution"),
            "num_frames": cfg_get(cfg, "model.num_frames"),
            "fps": cfg_get(cfg, "model.fps"),
            "block_index": cfg_get(cfg, "model.block_index"),
            "max_steps": cfg_get(cfg, "train.max_steps"),
            "num_steps": cfg_get(cfg, "inference.num_steps"),
            "cfg_scale": cfg_get(cfg, "inference.cfg_scale"),
        }
        values = {key: value for key, value in values.items() if value is not None}
        values.update(overrides)
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        """Fields of the configuration as a plain mapping."""
        return asdict(self)

    @property
    def width(self) -> int:
        """Frame width in pixels."""
        return int(self.resolution[0])

    @property
    def height(self) -> int:
        """Frame height in pixels."""
        return int(self.resolution[1])

    @property
    def latent_frames(self) -> int:
        """Temporal size of the cell grid and of the latent."""
        return latent_frames(self.num_frames)

    @property
    def grid(self) -> tuple[int, int, int]:
        """`(T, H, W)` of the backbone token grid, i.e. of the IGR annotations."""
        return latent_grid(self)

    @property
    def latent_shape(self) -> tuple[int, int, int, int]:
        """`(C, T, h, w)` of the video latent the VAE produces."""
        return latent_shape(self)


def latent_frames(num_frames: int, stride: int = TEMPORAL_STRIDE) -> int:
    """Temporal size of the cell grid for a clip of `num_frames` frames.

    Args:
        num_frames: Frames of the clip.
        stride: Frames covered by one temporal cell, 4 for the released backbone.

    Returns:
        `1 + (num_frames - 1) // stride`, e.g. 24 for the 93-frame protocol clip.
    """
    frames = int(num_frames)
    if frames < 1:
        raise ValueError(f"num_frames must be positive, got {num_frames}")
    return 1 + (frames - 1) // int(stride)


def latent_grid(config: Any = None) -> tuple[int, int, int]:
    """Cell grid of a clip, `(T, H, W)`, the grid the IGR annotations live on.

    Args:
        config: A :class:`GigaWorldConfig`, a mapping of its fields or a parsed run config;
            `None` uses the DreamGen preset.

    Returns:
        `(1 + (num_frames - 1) // 4, height // 16, width // 16)`, which is `(24, 30, 48)`
        for the DreamGen clip and `(24, 30, 40)` for the AgiBot one.
    """
    cfg = _as_config(config)
    return (latent_frames(cfg.num_frames), cfg.height // CELL_SIZE, cfg.width // CELL_SIZE)


def latent_shape(config: Any = None) -> tuple[int, int, int, int]:
    """Shape of the video latent, `(C, T, h, w)`.

    Args:
        config: A :class:`GigaWorldConfig`, a mapping of its fields or a parsed run config;
            `None` uses the DreamGen preset.

    Returns:
        `(16, 1 + (num_frames - 1) // 4, height // 8, width // 8)`, which is
        `(16, 24, 60, 96)` for the DreamGen clip.
    """
    cfg = _as_config(config)
    return (
        LATENT_CHANNELS,
        latent_frames(cfg.num_frames),
        cfg.height // VAE_SPATIAL_STRIDE,
        cfg.width // VAE_SPATIAL_STRIDE,
    )


def load_backbone(config: Any = None, **overrides: Any) -> Any:
    """Load the released GigaWorld-0 pipeline the training runs start from.

    The third-party checkout is put on `sys.path` before the import: `$GIGA_MODELS_DIR` and
    its parent when that directory is itself named `giga_models`, then
    `third_party/giga-world-0` of this repository, so no installation step is required
    beyond `scripts/setup/clone_gigaworld.sh`.

    Args:
        config: A :class:`GigaWorldConfig`, a mapping of its fields or a parsed run config;
            `None` uses the DreamGen preset.
        **overrides: Fields that take precedence over `config`.

    Returns:
        The pipeline object of the released package, loaded from the resolved checkpoint
        directory.

    Raises:
        ImportError: If `giga_models` cannot be imported.
        RuntimeError: If the checkpoint cannot be found, or if the released class exposes no
            `from_pretrained` classmethod.
    """
    cfg = _as_config(config, **overrides)
    _ensure_importable()
    try:
        from giga_models import GigaWorld0Pipeline  # noqa: PLC0415 - third-party checkout
    except ImportError as error:
        raise ImportError(
            f"the GigaWorld-0 backbone package {_BACKBONE_MODULE!r} is not importable "
            f"({error}); run scripts/setup/clone_gigaworld.sh, or point GIGA_MODELS_DIR at "
            "the directory holding the package; see docs/third_party.md"
        ) from error
    checkpoint = _resolve_checkpoint(cfg.checkpoint)
    builder = getattr(GigaWorld0Pipeline, "from_pretrained", None)
    if not callable(builder):
        raise RuntimeError(
            f"{_BACKBONE_CLASS} exposes no from_pretrained classmethod; load the released "
            f"pipeline manually and pass it to GigaWorldModel(pipeline=...); {_BACKBONE_CLASS} "
            f"carries {_public_names(GigaWorld0Pipeline)}"
        )
    logger.info("loading GigaWorld-0 checkpoint %s", checkpoint)
    return builder(str(checkpoint))


class GigaWorldModel:
    """EDM wrapper around a GigaWorld-0 pipeline.

    The wrapper adds the three things the trainer needs and the released pipeline does not
    carry as such: the latent encode/decode pair of its VAE, the EDM preconditioning of the
    restoration objective, and a parameter view of the trainable denoiser.

    Args:
        config: A :class:`GigaWorldConfig`, a mapping of its fields or a parsed run config;
            `None` uses the DreamGen preset.
        pipeline: The released pipeline; `None` loads it with :func:`load_backbone`.
        denoiser: The transformer to denoise with; `None` reads it off `pipeline`.
        vae: The video autoencoder; `None` reads it off `pipeline`.
        text_encoder: The text encoder; `None` reads it off `pipeline`.

    Attributes:
        config: The :class:`GigaWorldConfig` in use.
        pipeline: The released pipeline, loaded or passed in.
    """

    def __init__(
        self,
        config: Any = None,
        pipeline: Any = None,
        *,
        denoiser: Any = None,
        vae: Any = None,
        text_encoder: Any = None,
    ) -> None:
        self.config = _as_config(config)
        self.pipeline = load_backbone(self.config) if pipeline is None else pipeline
        self._denoiser = denoiser
        self._vae = vae
        self._text_encoder = text_encoder
        self._tokenizer: Any = None
        self._component_cache: dict[str, Any] = {}

    @property
    def grid(self) -> tuple[int, int, int]:
        """`(T, H, W)` of the backbone token grid, i.e. of the IGR annotations."""
        return latent_grid(self.config)

    @property
    def latent_shape(self) -> tuple[int, int, int, int]:
        """`(C, T, h, w)` of the video latent the VAE produces."""
        return latent_shape(self.config)

    @property
    def denoiser(self) -> Any:
        """The transformer whose blocks carry the TIA adapter."""
        return self._component("denoiser", self._denoiser, ("denoiser", "dit", "transformer", "model"))

    @property
    def backbone(self) -> Any:
        """Alias of :attr:`denoiser`, the module TIA hooks are registered on."""
        return self.denoiser

    @property
    def vae(self) -> Any:
        """The video autoencoder used by :meth:`encode_video` and :meth:`decode`."""
        return self._component("vae", self._vae, ("vae", "video_vae", "autoencoder"))

    @property
    def text_encoder(self) -> Any:
        """The text encoder used when the pipeline exposes no prompt encoder."""
        return self._component("text_encoder", self._text_encoder, ("text_encoder", "text_encoder_2"))

    @property
    def tokenizer(self) -> Any:
        """The tokenizer matching :attr:`text_encoder`."""
        return self._component("tokenizer", self._tokenizer, ("tokenizer", "text_tokenizer"))

    @property
    def device(self) -> torch.device:
        """Device of the trainable denoiser, `cpu` when it exposes no parameter."""
        for parameter in self._iter_parameters(self.denoiser):
            return parameter.device
        return torch.device("cpu")

    @property
    def dtype(self) -> torch.dtype:
        """Dtype of the trainable denoiser, `float32` when it exposes no parameter."""
        for parameter in self._iter_parameters(self.denoiser):
            return parameter.dtype
        return torch.float32

    def parameters(self) -> list[torch.nn.Parameter]:
        """Trainable parameters of the denoiser, what the optimiser steps over."""
        return [parameter for parameter in self._iter_parameters(self.denoiser) if parameter.requires_grad]

    def named_parameters(self) -> list[tuple[str, torch.nn.Parameter]]:
        """Named trainable parameters of the denoiser."""
        named = getattr(self.denoiser, "named_parameters", None)
        if not callable(named):
            return [(str(index), value) for index, value in enumerate(self.parameters())]
        return [(name, value) for name, value in named() if value.requires_grad]

    def train(self, mode: bool = True) -> GigaWorldModel:
        """Put the denoiser into training or evaluation mode."""
        self._set_mode(self.denoiser, mode)
        return self

    def eval(self) -> GigaWorldModel:
        """Put the denoiser into evaluation mode."""
        return self.train(False)

    def to(self, device: Any = None, dtype: Any = None) -> GigaWorldModel:
        """Move the denoiser, VAE and text encoder to a device and dtype."""
        vae = self._optional("vae", self._vae)
        text_encoder = self._optional("text_encoder", self._text_encoder)
        for component in (self.denoiser, vae, text_encoder):
            mover = getattr(component, "to", None)
            if not callable(mover):
                continue
            if device is not None and dtype is not None:
                mover(device=device, dtype=dtype)
            elif device is not None:
                mover(device)
            elif dtype is not None:
                mover(dtype)
        return self

    def encode_text(
        self, prompts: str | Sequence[str], negative_prompts: str | Sequence[str] | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Encode prompts into the conditioning the denoiser cross-attends to.

        Args:
            prompts: A prompt or a list of prompts, one per clip of the batch.
            negative_prompts: Optional negative prompts, same count as `prompts`, used for the
                classifier-free combination of :meth:`denoise`.

        Returns:
            The positive context and, when negatives were given, the negative context.

        Raises:
            RuntimeError: If the pipeline exposes neither a prompt encoder nor a text encoder
                with a tokenizer.
        """
        texts = _as_text_list(prompts)
        negatives = None if negative_prompts is None else _as_text_list(negative_prompts, len(texts))
        encoder = getattr(self.pipeline, "encode_prompt", None)
        if callable(encoder):
            positive = self._call_prompt_encoder(encoder, texts)
            negative = None if negatives is None else self._call_prompt_encoder(encoder, negatives)
            return positive, negative
        return self._encode_with_text_encoder(texts, negatives)

    def encode_video(self, video: Any) -> torch.Tensor:
        """Encode frames into the video latent the denoiser runs on.

        Args:
            video: `(T, H, W, 3)` uint8 frames, a `(C, T, H, W)` clip or a `(B, C, T, H, W)`
                batch, values in `[0, 255]`, `[0, 1]` or `[-1, 1]`.

        Returns:
            `(C, T, h, w)` latent for a single clip, `(B, C, T, h, w)` for a batch, on the
            device of the VAE.
        """
        tensor = _as_video_batch(video)
        latents = [_encode_clip(self.vae, clip.to(device=self._vae_device())) for clip in tensor]
        stacked = torch.stack(latents, dim=0)
        return stacked[0] if _is_single(video, tensor) else stacked

    def decode(self, latents: Any) -> torch.Tensor:
        """Decode latents back to a `(C, T, H, W)` frame tensor in `[-1, 1]`.

        Args:
            latents: `(C, T, h, w)` latent or a `(B, C, T, h, w)` batch.

        Returns:
            The decoded frames in the layout of the input's rank.
        """
        decoder = getattr(self.vae, "decode", None)
        if not callable(decoder):
            raise RuntimeError("the video VAE exposes no decode(); pass vae=... to GigaWorldModel")
        tensor = latents if torch.is_tensor(latents) else torch.as_tensor(latents)
        unbatched = tensor.ndim == 4
        if unbatched:
            tensor = tensor[None]
        with torch.no_grad():
            try:
                output = decoder(tensor, return_dict=False)
            except TypeError:
                output = decoder(tensor)
        frames = _first_tensor(output)
        return frames[0] if unbatched and frames.ndim == 5 else frames

    def sample_sigma(
        self,
        batch_size: int,
        *,
        device: Any = None,
        dtype: Any = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Draw the noise levels of a batch, `sigma = exp(p_mean + p_std * N(0, 1))`.

        Args:
            batch_size: Number of levels to draw.
            device: Device of the result; defaults to the denoiser's device.
            dtype: Dtype of the result; defaults to the denoiser's dtype.
            generator: Generator of the draw, for a reproducible run.

        Returns:
            `(batch_size,)` noise levels.
        """
        size = int(batch_size)
        if size < 1:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        noise = torch.randn(
            size, generator=generator, device=device or self.device, dtype=torch.float32
        )
        levels = torch.exp(EDM_P_MEAN + EDM_P_STD * noise)
        return levels.to(dtype=dtype or self.dtype)

    def denoise(
        self,
        latents: Any,
        context: Any,
        sigma: Any,
        *,
        cfg_scale: float = 7.0,
        negative_context: Any = None,
        fps: float | None = None,
        num_steps: int | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Denoise a latent at one or more noise levels, with the EDM preconditioning.

        With `num_steps` unset a single level is evaluated, the call the restoration loss
        trains through. With `num_steps` set the EDM sampler runs the `num_steps`-step
        schedule from `sigma_max` down to zero and returns a sampled latent, which is the
        local counterpart of the generation protocol the released pipeline implements in
        :meth:`inference`.

        Args:
            latents: `(C, T, h, w)` or `(B, C, T, h, w)` noisy latent.
            context: Positive conditioning, `(B, L, D)` or `(L, D)`.
            sigma: Noise level, a scalar or a `(B,)` tensor; ignored when sampling.
            cfg_scale: Classifier-free guidance scale, applied only when
                `negative_context` is given.
            negative_context: Negative conditioning, same layout as `context`.
            fps: Frame rate the backbone is conditioned on; `None` uses `config.fps`.
            num_steps: Sampling steps; `None` evaluates the levels of `sigma` in one step.
            generator: Generator of the initial noise of the sampling loop.

        Returns:
            `(B, C, T, h, w)` denoised or sampled latent, on the device of `latents`.
        """
        tensor = latents if torch.is_tensor(latents) else torch.as_tensor(latents)
        unbatched = tensor.ndim == 4
        if unbatched:
            tensor = tensor[None]
        rate = self.config.fps if fps is None else float(fps)
        if num_steps is not None:
            sampled = self._sample_latent(
                tensor.shape,
                context,
                num_steps=int(num_steps),
                cfg_scale=float(cfg_scale),
                negative_context=negative_context,
                fps=rate,
                device=tensor.device,
                dtype=tensor.dtype,
                generator=generator,
            )
            return sampled[0] if unbatched else sampled
        levels = _sigma_like(sigma, tensor.shape[0], tensor.device)
        estimate = self._denoise_at(
            tensor, context, levels, cfg_scale=float(cfg_scale), negative_context=negative_context, fps=rate
        )
        return estimate[0] if unbatched else estimate

    def forward(
        self,
        batch: Any,
        *,
        context: Any = None,
        negative_context: Any = None,
        generator: torch.Generator | None = None,
        fps: float | None = None,
    ) -> dict[str, Any]:
        """Run one restoration step of `eq:igr_construction` on a training batch.

        The clip is encoded to its clean latent, noise is drawn at a fresh level and the
        backbone predicts the clean latent from the noisy one, which is what `L_IGR`
        compares against.

        Args:
            batch: Sample or collated batch mapping. `video` is the disturbed clip, a
                `(C, T, H, W)` tensor or a `(B, C, T, H, W)` collated batch; `caption` or
                `instruction` carry the prompt when `context` is not passed; `sigma` may
                carry a level or a `(B,)` tensor of levels to reuse.
            context: Positive conditioning; `None` encodes the prompts of the batch.
            negative_context: Negative conditioning of the classifier-free combination;
                `None` leaves the prediction unconditional-free, as training needs.
            generator: Generator of the noise draw, for a reproducible run.
            fps: Frame rate the backbone is conditioned on; `None` uses `config.fps`.

        Returns:
            Mapping with the batch's `(B, C, T, h, w)` `pred` and clean `target` latent, its
            `weight_map`, the `(B,)` `sigma` of the draw and the `noise` added to the latent.
            A `(C, T, H, W)` clip is read as a batch of one, so every tensor is batched.
        """
        video = _batch_video(batch)
        target = self.encode_video(video)
        if target.ndim == 4:
            target = target[None]
        if context is None:
            context = self.encode_text(_batch_prompts(batch))[0]
        rate = self.config.fps if fps is None else float(fps)
        sigma = _batch_sigma(batch, target.shape[0], target.device) or self.sample_sigma(
            target.shape[0], device=target.device, dtype=target.dtype, generator=generator
        )
        noise = torch.randn(target.shape, generator=generator, device=target.device, dtype=target.dtype)
        noisy = target + sigma.view(-1, 1, 1, 1, 1).to(target.dtype) * noise
        pred = self._denoise_at(
            noisy,
            context,
            sigma,
            cfg_scale=1.0,
            negative_context=negative_context,
            fps=rate,
        )
        return {
            "pred": pred,
            "target": target,
            "weight_map": _batch_weight_map(batch),
            "sigma": sigma,
            "noise": noise,
        }

    def igr_loss(
        self, pred: Any, target: Any, weight_map: Any, sigma: Any, *, sigma_data: float = SIGMA_DATA
    ) -> torch.Tensor:
        """The `L_IGR` restoration term of `eq:pipeline` for one batch.

        The weight map lives on the `16 px` cell grid of the annotations while the latent
        grid is twice as fine, which the term handles itself with
        :func:`eveworld.data.transforms.latent.resize_weight_map`; its unit mean, and with it
        the loss scale, is preserved by the resize.

        Args:
            pred: `(B, C, T, h, w)` denoised latent of the disturbed clip.
            target: Clean latent of the demonstration, same shape as `pred`.
            weight_map: `(B, T, H, W)` or `(T, H, W)` cell-grid map of the sample. A batched
                map is read one per member, so every member is weighted by its own map.
            sigma: `(B,)` noise levels of the batch.
            sigma_data: Standard deviation of the training data.

        Returns:
            Scalar loss tensor.
        """
        from eveworld.methods.igr.loss import igr_loss

        return igr_loss(
            pred,
            target,
            _weight_map_for(pred, weight_map),
            sigma,
            sigma_data=sigma_data,
        )

    def inference(
        self,
        prompt: str | Sequence[str],
        *,
        negative_prompt: str | Sequence[str] | None = None,
        num_frames: int | None = None,
        num_steps: int | None = None,
        cfg_scale: float | None = None,
        seed: int | None = None,
    ) -> np.ndarray:
        """Generate a clip with the sampling loop of the released pipeline.

        Generation is delegated to the pipeline rather than reimplemented, so the released
        scheduler, its sigma shift and its conditioning stay exactly as published; only the
        arguments the callable accepts are passed, read from its signature.

        Args:
            prompt: Prompt of the clip.
            negative_prompt: Optional negative prompt of the classifier-free combination.
            num_frames: Frames to generate; `None` uses `config.num_frames`.
            num_steps: Sampling steps; `None` uses `config.num_steps`.
            cfg_scale: Guidance scale; `None` uses `config.cfg_scale`.
            seed: Seed of the sampler; `None` leaves the pipeline's own default.

        Returns:
            `(T, H, W, 3)` uint8 RGB frames.

        Raises:
            RuntimeError: If the pipeline exposes no sampling callable.
        """
        callable_ = _generator_callable(self.pipeline)
        frames = self.config.num_frames if num_frames is None else int(num_frames)
        steps = self.config.num_steps if num_steps is None else int(num_steps)
        scale = self.config.cfg_scale if cfg_scale is None else float(cfg_scale)
        texts = _as_text_list(prompt)
        wants = (
            ("prompt", texts if len(texts) > 1 else texts[0]),
            ("negative_prompt", negative_prompt),
            ("num_frames", frames),
            ("num_inference_steps", steps),
            ("steps", steps),
            ("guidance_scale", scale),
            ("cfg_scale", scale),
            ("fps", self.config.fps),
            ("height", self.config.height),
            ("width", self.config.width),
            ("seed", seed),
            ("device", self.device),
        )
        accepted = _signature_names(callable_)
        free = "**kwargs" in accepted
        kwargs = {
            name: value
            for name, value in wants
            if value is not None and (free or name in accepted)
        }
        logger.info("sampling %d frames in %d steps at cfg %.1f", frames, steps, scale)
        output = callable_(**kwargs)
        return _as_frames(output)

    def _sample_latent(
        self,
        shape: Sequence[int],
        context: Any,
        *,
        num_steps: int,
        cfg_scale: float,
        negative_context: Any,
        fps: float,
        device: torch.device,
        dtype: torch.dtype,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        """EDM sampling loop over the `rho`-schedule from `sigma_max` down to zero."""
        if num_steps < 1:
            raise ValueError(f"num_steps must be positive, got {num_steps}")
        batch = torch.zeros(shape, device=device, dtype=dtype)
        noise = torch.randn(batch.shape, generator=generator, device=device, dtype=dtype)
        batch = batch + noise * _sigma_max()
        schedule = _edm_schedule(num_steps).to(device=device, dtype=dtype)
        for level, following in zip(schedule[:-1], schedule[1:]):
            estimate = self._denoise_at(
                batch,
                context,
                level.reshape(1),
                cfg_scale=cfg_scale,
                negative_context=negative_context,
                fps=fps,
            )
            derivative = (batch - estimate) / level.clamp_min(1e-8)
            batch = batch + derivative * (following - level)
        return batch

    def _denoise_at(
        self,
        latents: torch.Tensor,
        context: Any,
        levels: torch.Tensor,
        *,
        cfg_scale: float,
        negative_context: Any,
        fps: float,
    ) -> torch.Tensor:
        """One EDM evaluation at the given levels, with the classifier-free combination."""
        coefficient = _preconditioning(levels, latents.dtype)
        noisy = latents * coefficient["c_in"].view(-1, 1, 1, 1, 1)
        raw = self._call_denoiser(noisy, coefficient["c_noise"], context, fps=fps)
        if negative_context is not None and float(cfg_scale) != 1.0:
            unconditional = self._call_denoiser(noisy, coefficient["c_noise"], negative_context, fps=fps)
            raw = unconditional + float(cfg_scale) * (raw - unconditional)
        c_skip = coefficient["c_skip"].view(-1, 1, 1, 1, 1)
        c_out = coefficient["c_out"].view(-1, 1, 1, 1, 1)
        return c_skip * latents + c_out * raw

    def _call_denoiser(
        self, latents: torch.Tensor, levels: torch.Tensor, context: Any, *, fps: float | None
    ) -> torch.Tensor:
        """Call the denoiser with the argument names of its own signature.

        Raises:
            RuntimeError: If the signature carries none of the accepted names for the latent,
                the level or the conditioning.
        """
        denoiser = self.denoiser
        target = denoiser.forward if callable(getattr(denoiser, "forward", None)) else denoiser
        accepted = _signature_names(target)
        free = "**kwargs" in accepted
        latent_name = _first_name(accepted, ("hidden_states", "x", "latents", "inputs", "sample"), free)
        level_name = _first_name(accepted, ("timesteps", "timestep", "t", "sigma", "noise_labels"), free)
        context_name = _first_name(accepted, ("encoder_hidden_states", "crossattn_emb", "context"), free)
        if latent_name is None or level_name is None or context_name is None:
            raise RuntimeError(
                "cannot call the GigaWorld-0 denoiser: expected arguments named like "
                "hidden_states/x/latents, timesteps/timestep/t, encoder_hidden_states/crossattn_emb, "
                f"found {sorted(accepted)}"
            )
        kwargs: dict[str, Any] = {latent_name: latents, level_name: levels, context_name: context}
        if fps is not None and (free or "fps" in accepted):
            kwargs["fps"] = torch.as_tensor(int(round(float(fps))), device=latents.device)
        if "padding_mask" in accepted:
            mask_shape = (
                latents.shape[0],
                1,
                latents.shape[-2] * VAE_SPATIAL_STRIDE,
                latents.shape[-1] * VAE_SPATIAL_STRIDE,
            )
            kwargs["padding_mask"] = torch.zeros(mask_shape, device=latents.device, dtype=latents.dtype)
        output = target(**kwargs)
        return _first_tensor(output)

    def _call_prompt_encoder(self, encoder: Any, texts: Sequence[str]) -> torch.Tensor:
        """Call the pipeline's prompt encoder with the arguments its signature accepts."""
        accepted = _signature_names(encoder)
        kwargs: dict[str, Any] = {}
        if "device" in accepted:
            kwargs["device"] = self.device
        if "num_images_per_prompt" in accepted:
            kwargs["num_images_per_prompt"] = 1
        if "do_classifier_free_guidance" in accepted:
            kwargs["do_classifier_free_guidance"] = False
        try:
            output = encoder(list(texts), **kwargs)
        except TypeError:
            output = encoder(list(texts))
        return _first_tensor(output)

    def _encode_with_text_encoder(
        self, texts: Sequence[str], negatives: Sequence[str] | None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Encode prompts with the pipeline's own text encoder and tokenizer."""
        encoder = self._optional("text_encoder", self._text_encoder)
        tokenizer = self._optional("tokenizer", self._tokenizer)
        if encoder is None or tokenizer is None:
            raise RuntimeError(
                "the GigaWorld-0 pipeline exposes neither encode_prompt() nor a text_encoder "
                f"with a tokenizer; it carries {_public_names(self.pipeline)}"
            )
        positive = _encode_with_tokenizer(encoder, tokenizer, texts, self.device)
        if negatives is None:
            negative = None
        else:
            negative = _encode_with_tokenizer(encoder, tokenizer, negatives, self.device)
        return positive, negative

    def _component(self, name: str, explicit: Any, candidates: Sequence[str]) -> Any:
        """Find a component of the pipeline once, under one of its known names."""
        if explicit is not None:
            return explicit
        if name in self._component_cache:
            return self._component_cache[name]
        for candidate in candidates:
            value = getattr(self.pipeline, candidate, None)
            if value is not None:
                self._component_cache[name] = value
                return value
        raise RuntimeError(
            f"the GigaWorld-0 pipeline exposes no {name}; expected one of {list(candidates)} and "
            f"found {_public_names(self.pipeline)}; pass {name}=... to GigaWorldModel"
        )

    def _optional(self, name: str, explicit: Any) -> Any:
        """Component lookup that returns `None` instead of raising."""
        try:
            return self._component(name, explicit, _COMPONENT_NAMES[name])
        except RuntimeError:
            return None

    def _vae_device(self) -> torch.device:
        """Device of the VAE parameters, `cpu` when it exposes none."""
        for parameter in self._iter_parameters(self.vae):
            return parameter.device
        return torch.device("cpu")

    @staticmethod
    def _iter_parameters(module: Any):
        """Iterate over the parameters of a module-like object, empty when it has none."""
        parameters = getattr(module, "parameters", None)
        if not callable(parameters):
            return
        for parameter in parameters():
            if torch.is_tensor(parameter):
                yield parameter

    @staticmethod
    def _set_mode(module: Any, mode: bool) -> None:
        """Put a module-like object into training or evaluation mode."""
        switch = getattr(module, "train", None)
        if callable(switch):
            switch(bool(mode))


_COMPONENT_NAMES: dict[str, tuple[str, ...]] = {
    "denoiser": ("denoiser", "dit", "transformer", "model"),
    "vae": ("vae", "video_vae", "autoencoder"),
    "text_encoder": ("text_encoder", "text_encoder_2"),
    "tokenizer": ("tokenizer", "text_tokenizer"),
}


def _infer_variant(cfg: Any) -> str:
    """Variant named by `model.variant`, or read off the run name, split and metadata."""
    declared = cfg_get(cfg, "model.variant")
    if declared is not None:
        return str(declared)
    haystack = " ".join(
        str(cfg_get(cfg, key, "") or "") for key in ("name", "data.split", "data.metadata", "output_dir")
    ).lower()
    return "agibot" if "agibot" in haystack else "dreamgen"


def _as_config(config: Any = None, **overrides: Any) -> GigaWorldConfig:
    """Normalise the `config` argument of the public entry points.

    Accepts a :class:`GigaWorldConfig`, a mapping of its fields, a parsed run config carrying
    the `model` block, or `None` for the DreamGen preset.
    """
    if config is None:
        base = GigaWorldConfig()
    elif isinstance(config, GigaWorldConfig):
        base = config
    elif isinstance(config, Mapping):
        keys = {str(key) for key in config}
        if keys & {"model", "train", "inference", "method", "data"}:
            base = GigaWorldConfig.from_config(config)
        else:
            base = GigaWorldConfig(**{str(key): value for key, value in config.items()})
    else:
        raise TypeError(
            f"config must be a GigaWorldConfig, a mapping or None, got {type(config).__name__}"
        )
    return replace(base, **overrides) if overrides else base


def _as_resolution(value: Any) -> tuple[int, int]:
    """Read a `(width, height)` pair and check it against the cell size."""
    if value is None:
        raise ValueError("resolution must be given as a (width, height) pair")
    try:
        width, height = int(value[0]), int(value[1])
    except (TypeError, IndexError, KeyError) as error:
        raise ValueError(f"resolution must be a (width, height) pair, got {value!r}") from error
    if width <= 0 or height <= 0:
        raise ValueError(f"resolution must be positive, got {(width, height)}")
    if width % CELL_SIZE or height % CELL_SIZE:
        raise ValueError(
            f"resolution {(width, height)} must be a multiple of the {CELL_SIZE} px cell size "
            "of the backbone token grid"
        )
    return width, height


def _ensure_importable() -> None:
    """Put the third-party GigaWorld-0 checkout on `sys.path`."""
    candidates: list[Path] = []
    configured = os.environ.get("GIGA_MODELS_DIR")
    if configured:
        directory = Path(configured).expanduser()
        candidates.append(directory)
        if directory.name == _BACKBONE_MODULE:
            candidates.append(directory.parent)
    candidates.append(repo_root() / _THIRD_PARTY_CHECKOUT)
    for directory in reversed([path for path in candidates if path.is_dir()]):
        text = str(directory)
        if text in sys.path:
            sys.path.remove(text)
        sys.path.insert(0, text)


def _checkpoint_roots() -> list[Path]:
    """Directories a released checkpoint is looked up under, in order."""
    roots: list[Path] = []
    for variable in ("GW0_MODEL_DIR", "EVEWORLD_CHECKPOINT_ROOT"):
        value = os.environ.get(variable)
        if value:
            roots.append(Path(value).expanduser())
    roots.append(repo_root() / "checkpoints")
    return roots


def _resolve_checkpoint(name: str | None) -> Path:
    """Resolve the checkpoint directory of a released GigaWorld-0 name.

    Raises:
        RuntimeError: If no candidate directory exists, listing every path that was tried.
    """
    value = DEFAULT_MODEL_NAME if name is None or not str(name).strip() else str(name).strip()
    explicit = Path(value).expanduser()
    tried: list[Path] = []
    for candidate in _checkpoint_candidates(explicit, value):
        if candidate in tried:
            continue
        tried.append(candidate)
        if candidate.is_dir():
            return candidate.resolve()
    listing = "\n".join(f"  - {path}" for path in tried)
    raise RuntimeError(
        f"GigaWorld-0 checkpoint {value!r} was not found; tried:\n{listing}\n"
        "download it with scripts/setup/download_models.sh (which honours "
        "EVEWORLD_CHECKPOINT_ROOT), point GW0_MODEL_DIR at it, or pass an explicit directory; "
        "see checkpoints/README.md"
    )


def _checkpoint_candidates(explicit: Path, value: str) -> list[Path]:
    """Every path a checkpoint name is resolved against, most specific first."""
    candidates = [explicit]
    for root in _checkpoint_roots():
        candidates.append(root / explicit)
        if explicit.name != explicit.as_posix():
            candidates.append(root / explicit.name)
        if root.name == explicit.name or root.name.lower() == value.lower():
            candidates.append(root)
    return candidates


def _as_video_batch(video: Any) -> torch.Tensor:
    """Read frames as a `(B, C, T, H, W)` float32 tensor in `[-1, 1]`."""
    if torch.is_tensor(video):
        tensor = video.detach().to(dtype=torch.float32)
    else:
        array = np.asarray(video, dtype=np.float32)
        tensor = torch.from_numpy(np.ascontiguousarray(array))
    if tensor.ndim == 4 and tensor.shape[-1] == 3:
        tensor = tensor.permute(3, 0, 1, 2)[None]
    elif tensor.ndim == 4:
        tensor = tensor[None]
    elif tensor.ndim == 5 and tensor.shape[-1] == 3:
        tensor = tensor.permute(0, 4, 1, 2, 3)
    if tensor.ndim != 5 or tensor.shape[1] != 3:
        raise ValueError(
            f"Expected (T, H, W, 3) frames, a (C, T, H, W) clip or a batch, got {tuple(tensor.shape)}"
        )
    return _to_unit_range(tensor)


def _to_unit_range(tensor: torch.Tensor) -> torch.Tensor:
    """Map frames to `[-1, 1]`, the range the video VAE was trained on."""
    if float(tensor.min()) >= 0.0 and float(tensor.max()) > 1.5:
        return tensor / 127.5 - 1.0
    if float(tensor.min()) >= 0.0:
        return tensor * 2.0 - 1.0
    return tensor


def _is_single(video: Any, batched: torch.Tensor) -> bool:
    """Whether the caller handed in a single clip rather than a batch."""
    return not (torch.is_tensor(video) and batched.shape[0] > 0 and _was_batched(video))


def _was_batched(video: Any) -> bool:
    """Whether a tensor input carried a leading batch dimension."""
    tensor = video if torch.is_tensor(video) else torch.as_tensor(video)
    return tensor.ndim == 5


def _encode_clip(vae: Any, clip: torch.Tensor) -> torch.Tensor:
    """Encode one `(C, T, H, W)` clip, keeping it on the device of the VAE.

    `eveworld.data.transforms.latent.encode_latent` moves its input to the CPU, which a
    training step on a GPU cannot use, so the encoder is called directly here.
    """
    encoder = getattr(vae, "encode", None) or getattr(vae, "encode_", None)
    if not callable(encoder):
        raise RuntimeError(
            "the video VAE exposes neither encode() nor encode_(); pass vae=... to GigaWorldModel"
        )
    with torch.no_grad():
        try:
            output = encoder(clip[None], return_dict=False)
        except TypeError:
            output = encoder(clip[None])
    latent = _first_tensor(output)
    return latent[0] if latent.ndim == 5 else latent


def _first_tensor(output: Any) -> torch.Tensor:
    """Read a tensor out of the many shapes a pipeline output arrives in."""
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)):
        for item in output:
            if torch.is_tensor(item):
                return item
            hidden = getattr(item, "last_hidden_state", None)
            if torch.is_tensor(hidden):
                return hidden
        raise TypeError(f"no tensor in the {type(output).__name__} returned by the pipeline")
    distribution = getattr(output, "latent_dist", None)
    if distribution is not None:
        for name in ("mode", "mean", "sample"):
            value = getattr(distribution, name, None)
            if callable(value):
                return value()
            if torch.is_tensor(value):
                return value
    for name in ("sample", "last_hidden_state", "frames", "video", "videos"):
        value = getattr(output, name, None)
        if torch.is_tensor(value):
            return value
    raise TypeError(f"cannot read a tensor out of {type(output).__name__}")


def _as_frames(output: Any) -> np.ndarray:
    """Turn a generation result into `(T, H, W, 3)` uint8 frames."""
    tensor = _first_tensor(output)
    while tensor.ndim > 4:
        tensor = tensor[0]
    if tensor.ndim != 4:
        raise ValueError(f"Expected generated frames, got {tuple(tensor.shape)}")
    array = tensor.detach().to(device="cpu", dtype=torch.float32).numpy()
    if array.shape[-1] == 3:
        array = array.transpose(0, 3, 1, 2)
    elif array.shape[0] != 3:
        raise ValueError(f"Expected generated frames, got {tuple(tensor.shape)}")
    if float(array.min()) < 0.0:
        array = (array + 1.0) * 0.5
    array = np.clip(array, 0.0, 1.0)
    return np.round(array * 255.0).astype(np.uint8).transpose(1, 2, 3, 0)


def _encode_with_tokenizer(
    encoder: Any, tokenizer: Any, texts: Sequence[str], device: torch.device
) -> torch.Tensor:
    """Encode prompts with a tokenizer and a text encoder."""
    accepted = _signature_names(tokenizer)
    kwargs: dict[str, Any] = {"return_tensors": "pt"}
    if "padding" in accepted:
        kwargs["padding"] = "max_length"
    if "truncation" in accepted:
        kwargs["truncation"] = True
    batch = tokenizer(list(texts), **kwargs)
    inputs: dict[str, Any] = {}
    for key, value in dict(batch).items():
        inputs[key] = value.to(device) if torch.is_tensor(value) else value
    with torch.no_grad():
        output = encoder(**inputs)
    return _first_tensor(output)


def _generator_callable(pipeline: Any) -> Any:
    """The sampling callable of a pipeline, under any of its known names."""
    for name in ("generate", "infer", "sample", "generate_video", "inference"):
        candidate = getattr(pipeline, name, None)
        if callable(candidate):
            return candidate
    if callable(pipeline):
        return pipeline
    raise RuntimeError(
        f"the GigaWorld-0 pipeline exposes no sampling callable; it carries {_public_names(pipeline)}"
    )


def _signature_names(function: Any) -> frozenset[str]:
    """Parameter names of a callable, `**kwargs` reported as `"**kwargs"`."""
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return frozenset()
    names = set()
    for name, parameter in signature.parameters.items():
        names.add("**kwargs" if parameter.kind is inspect.Parameter.VAR_KEYWORD else name)
    return frozenset(names)


def _first_name(accepted: frozenset[str], options: Sequence[str], free: bool) -> str | None:
    """First of `options` the signature carries; `hidden_states` under `**kwargs`."""
    if free:
        return options[0]
    for option in options:
        if option in accepted:
            return option
    return None


def _preconditioning(levels: torch.Tensor, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    """EDM coefficients of `docs/training.md` at the given noise levels."""
    sigma = levels.to(dtype=torch.float32)
    data = torch.as_tensor(SIGMA_DATA, dtype=torch.float32, device=sigma.device)
    total = sigma * sigma + data * data
    root = torch.sqrt(total)
    return {
        "c_in": (1.0 / root).to(dtype),
        "c_skip": (data * data / total).to(dtype),
        "c_out": (sigma * data / root).to(dtype),
        "c_noise": (torch.log(sigma) / 4.0).to(dtype),
    }


def _edm_schedule(
    num_steps: int, *, sigma_min: float = 0.002, sigma_max: float = 80.0, rho: float = 7.0
) -> torch.Tensor:
    """EDM noise schedule, `num_steps + 1` levels from `sigma_max` down to zero."""
    steps = torch.arange(int(num_steps), dtype=torch.float64)
    ramp = steps / max(int(num_steps) - 1, 1)
    levels = (
        sigma_max ** (1.0 / rho) + ramp * (sigma_min ** (1.0 / rho) - sigma_max ** (1.0 / rho))
    ) ** rho
    return torch.cat((levels, torch.zeros(1, dtype=torch.float64))).to(torch.float32)


def _sigma_max() -> torch.Tensor:
    """Initial noise scale of the EDM sampler."""
    return torch.as_tensor(80.0, dtype=torch.float32)


def _sigma_like(sigma: Any, batch_size: int, device: torch.device) -> torch.Tensor:
    """Broadcast a scalar or `(B,)` level onto a batch."""
    levels = sigma if torch.is_tensor(sigma) else torch.as_tensor(sigma, dtype=torch.float32)
    levels = levels.detach().to(device=device, dtype=torch.float32).reshape(-1)
    if levels.numel() == 1:
        levels = levels.expand(batch_size)
    if levels.numel() != batch_size:
        raise ValueError(f"sigma holds {levels.numel()} levels for a batch of {batch_size}")
    if float(levels.min()) <= 0.0:
        raise ValueError("sigma must be positive")
    return levels


def _batch_video(batch: Any) -> Any:
    """Read the clip out of a sample or a collated batch."""
    if isinstance(batch, Mapping):
        if "video" not in batch:
            raise KeyError(f"the batch carries no 'video' entry; it has {sorted(batch)}")
        return batch["video"]
    return batch


def _batch_prompts(batch: Any) -> list[str]:
    """Read the prompts of a batch, preferring the instruction over the caption."""
    if not isinstance(batch, Mapping):
        raise ValueError("a prompt is needed when context is not passed explicitly")
    for key in ("instruction", "caption", "prompt"):
        value = batch.get(key)
        if value is None:
            continue
        return _as_text_list(value)
    raise KeyError(f"the batch carries no 'instruction', 'caption' or 'prompt'; it has {sorted(batch)}")


def _batch_weight_map(batch: Any) -> Any:
    """Read the optional weight map of a batch."""
    return batch.get("weight_map") if isinstance(batch, Mapping) else None


def _weight_map_for(pred: torch.Tensor, weight_map: Any) -> Any:
    """Read a collated `(B, T, h, w)` map as one map per member, `(B, 1, T, h, w)`.

    :func:`eveworld.methods.igr.loss.igr_loss` broadcasts a map of `(h, w)`, `(T, h, w)`,
    `(C, T, h, w)` or `(B, C, T, h, w)`, while the collation of the datasets stacks the
    per-sample `(T, h, w)` maps of the cell grid into `(B, T, h, w)`. The leading dimension is
    read as the batch one when it matches `pred`, which gives every member of the batch its own
    map; a map of another rank is passed through unchanged.
    """
    if weight_map is None:
        return None
    shape = tuple(int(value) for value in np.shape(weight_map))
    if len(shape) == 4 and shape[0] == int(pred.shape[0]):
        if torch.is_tensor(weight_map):
            return weight_map.unsqueeze(1)
        return np.expand_dims(weight_map, 1)
    return weight_map


def _batch_sigma(batch: Any, batch_size: int, device: torch.device) -> torch.Tensor | None:
    """Read the optional noise levels of a batch."""
    if not isinstance(batch, Mapping) or batch.get("sigma") is None:
        return None
    value = batch["sigma"]
    if torch.is_tensor(value) and value.numel() == 0:
        return None
    return _sigma_like(value, batch_size, device)


def _as_text_list(prompts: Any, expected: int | None = None) -> list[str]:
    """Read one or more prompts as a list of strings."""
    if isinstance(prompts, str):
        texts = [prompts]
    else:
        texts = [str(item) for item in prompts]
    if not texts:
        raise ValueError("at least one prompt is required")
    if expected is not None and len(texts) == 1 and expected > 1:
        texts = texts * expected
    if expected is not None and len(texts) != expected:
        raise ValueError(f"expected {expected} prompts, got {len(texts)}")
    return texts


def _public_names(obj: Any) -> list[str]:
    """Public attribute names of an object, for the error messages of the lookups."""
    return sorted(name for name in dir(obj) if not name.startswith("_"))
