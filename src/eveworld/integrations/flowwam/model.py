"""FlowWAM as the EVEWorld manipulation backbone.

Three things live here: the geometry of the backbone, the :class:`FlowWAMConfig` that
mirrors the `model` block of `configs/paper/flowwam/robotwin/eveworld.yaml`, and
:class:`FlowWAMModel`, the wrapper the trainer and the evaluation scripts talk to.

Geometry. The Wan2.2 5B video VAE the release ships compresses four frames into one
latent frame and 16 px into one latent cell, and its latent carries 48 channels, so a
`480x640` clip of 29 frames becomes a `(48, 8, 30, 40)` latent. The DiT folds the latent
into `2x2` patches, which puts 300 tokens per frame on a `15x20` grid; that is the grid
TIA is attached to and the grid the TIA adapter reshapes the block activations onto.
The IGR weight map lives on the same 16 px cells as the latent, so
:func:`latent_grid` returns `(8, 30, 40)` and :func:`latent_shape` the `(48, 8, 30, 40)`
latent.

Objective. `eveworld/flowwam_port/eve_flowwam_train.py` trains with flow matching: the
RGB stream and the optical-flow stream are noised with the same shifted-linear sigma
schedule, the network predicts the velocity `noise - clean` and the loss compares it
against that target under the IGR weight map, with the first latent frame masked because
it is pinned to the conditioning frame. A second head predicts the clean latent, which
sampling integrates with an Euler step over the same schedule. TIA adds an InfoNCE term
between the block-12 tokens of the two streams.

Two streams. The released pipeline carries a `WanVideoPipeline` and a `FlowStreamModule`
side by side; `diffsynth.pipelines.wan_video_dual_stream` runs both through the DiT
blocks with joint self-attention, so every evaluation of the denoiser consumes two
latents and returns two velocities. A separate module function conditions the flow
stream on the *rendered* robot-only video instead of sampling it, which is the
`--flow-cond robot_only` mode of `inference/arm_generate.py`.

The wrapper is duck-typed on purpose: the released pipeline exposes its denoiser, VAE,
flow module and prompt encoder under names that differ between releases, so the lookups
try the known spellings and raise a `RuntimeError` listing what the object does carry
instead of failing with an `AttributeError` deep inside a sampling loop. The third-party
packages are imported inside the functions that need them, so this module imports
without `diffsynth`, `peft` or a GPU present.
"""

from __future__ import annotations

import importlib.util
import inspect
import math
import os
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch

from eveworld.utils.config import cfg_get
from eveworld.utils.io import repo_root
from eveworld.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "CELL_SIZE",
    "DEFAULT_BASE_MODEL",
    "DEFAULT_LORA_RANK",
    "DEFAULT_MODEL_NAME",
    "LATENT_CHANNELS",
    "PATCH_SIZE",
    "TEMPORAL_STRIDE",
    "VAE_SPATIAL_STRIDE",
    "FlowWAMConfig",
    "FlowWAMModel",
    "latent_frames",
    "latent_grid",
    "latent_shape",
    "load_backbone",
]


CELL_SIZE = 16
"""Side of a cell of the latent grid in pixels, the grid the IGR weight map lives on."""

TEMPORAL_STRIDE = 4
"""Frames covered by one temporal cell of the latent, the Wan2.2 VAE compression."""

VAE_SPATIAL_STRIDE = 16
"""Spatial compression of the video VAE, the ratio between the frames and the latent."""

LATENT_CHANNELS = 48
"""Channels of the video latent, 48 for the Wan2.2 5B VAE the backbone ships."""

PATCH_SIZE = 2
"""Spatial patch the DiT folds the latent with, `(h/2)*(w/2)` tokens per frame."""

DEFAULT_MODEL_NAME = "FlowWAM-Stage-1"
"""Released FlowWAM checkpoint named by `model.checkpoint` in every configuration."""

DEFAULT_BASE_MODEL = "Wan2.2-TI2V-5B"
"""Wan base checkpoint the released fine-tuning starts from."""

DEFAULT_LORA_RANK = 32
"""Rank of the LoRA adapters the released fine-tuning protocol injects."""

_BACKBONE_MODULE = "diffsynth"
_BACKBONE_CLASS = "WanVideoPipeline"
_LOADER_MODULE = "pipeline_loader"
_THIRD_PARTY_CHECKOUT = Path("third_party") / "FlowWAM"
_TIMESTEP_SCALE = 1000.0
_DEFAULT_SEED = 42
_CROSSFADE = 10
_WINDOW_FRAMES = 121
_WINDOW_STRIDE = _WINDOW_FRAMES - _CROSSFADE - 1

_WEIGHT_ALIASES: dict[str, str] = {
    "flowwam_stage_1": "flowwam/flowwam_worldarena_stage1.safetensors",
    "flowwam_stage1": "flowwam/flowwam_worldarena_stage1.safetensors",
    "flowwam_stage_1_safetensors": "flowwam/flowwam_worldarena_stage1.safetensors",
    "stage1": "flowwam/flowwam_worldarena_stage1.safetensors",
    "stage_1": "flowwam/flowwam_worldarena_stage1.safetensors",
    "robotwin": "flowwam/flowwam_robotwin.safetensors",
    "flowwam_robotwin": "flowwam/flowwam_robotwin.safetensors",
}

_BASE_ALIASES: dict[str, str] = {
    "wan2_2_ti2v_5b": "wan2.2-ti2v-5b",
    "wan22_ti2v_5b": "wan2.2-ti2v-5b",
    "ti2v_5b": "wan2.2-ti2v-5b",
    "wan2_2": "wan2.2-ti2v-5b",
}

_VARIANT_PRESETS: dict[str, dict[str, Any]] = {
    "robotwin": {
        "checkpoint": DEFAULT_MODEL_NAME,
        "base_model": DEFAULT_BASE_MODEL,
        "lora_rank": DEFAULT_LORA_RANK,
        "resolution": (640, 480),
        "num_frames": 29,
        "fps": 24.0,
        "block_index": 12,
        "max_steps": 1128,
    },
}


@dataclass
class FlowWAMConfig:
    """Geometry, checkpoint and generation protocol of a FlowWAM run.

    Fields left at `None` are filled from the preset of `variant`, so `FlowWAMConfig()`
    carries the `640x480`, 29-frame RoboTwin protocol of
    `configs/paper/flowwam/robotwin/eveworld.yaml`.

    Attributes:
        variant: Name of the preset that fills the unset fields, `"robotwin"` for the only
            released transfer experiment.
        backbone: Name of the backbone family, `"flowwam"` in every released config.
        checkpoint: Name, file or directory of the fine-tuned weights, resolved by
            :func:`load_backbone` under `FLOWWAM_MODEL_DIR`, `EVEWORLD_CHECKPOINT_ROOT` or
            `checkpoints/`. `"FlowWAM-Stage-1"` maps to the released WorldArena weights.
        base_model: Name of the Wan base checkpoint the release is built on, resolved to
            `wan2.2-ti2v-5b` under the same roots.
        base_path: Directory of the base checkpoint; `None` resolves `base_model`.
        lora_rank: Rank of the LoRA adapters the training run injects.
        resolution: `(width, height)` of the frames in pixels, both a multiple of
            :data:`CELL_SIZE` times :data:`PATCH_SIZE`.
        num_frames: Frames of a protocol clip, 29 for RoboTwin and 121 for a full window.
        fps: Frame rate the backbone is conditioned on, 24 for RoboTwin.
        block_index: Transformer block TIA is attached to, 12 for RoboTwin.
        max_steps: Optimisation steps of the released run, 1128 for RoboTwin.
        num_steps: Sampling steps of the released generation protocol.
        cfg_scale: Classifier-free guidance scale of the released protocol, 1.0 for the
            released robot-only runs.
        sigma_shift: Shift of the flow-matching noise schedule, 5.0 in the released runs.
        flow_cond: `"robot_only"` conditions the flow stream on a rendered robot-only
            video, `"none"` samples it jointly with the RGB stream from noise.
        tia_inject: Whether generation injects the registered TIA adapter.
        full_traj: `"direct"` generates the whole clip in one pass, `"on"` rolls
            121-frame windows with a cross-faded overlap; `"off"` means `"direct"`.

    Raises:
        ValueError: If the variant is unknown, the resolution is not a positive multiple of
            32 px, or a frame count, frame rate, block index, step budget or rank is out
            of range.
    """

    variant: str = "robotwin"
    backbone: str = "flowwam"
    checkpoint: str | None = None
    base_model: str | None = None
    base_path: str | None = None
    lora_rank: int | None = None
    resolution: tuple[int, int] | None = None
    num_frames: int | None = None
    fps: float | None = None
    block_index: int | None = None
    max_steps: int | None = None
    num_steps: int = 40
    cfg_scale: float = 5.0
    sigma_shift: float = 5.0
    flow_cond: Any = "robot_only"
    tia_inject: Any = False
    full_traj: Any = "direct"

    def __post_init__(self) -> None:
        self.variant = str(self.variant).strip().lower()
        if self.variant not in _VARIANT_PRESETS:
            raise ValueError(f"Unknown FlowWAM variant {self.variant!r}; expected one of " f"{sorted(_VARIANT_PRESETS)}")
        for name, value in _VARIANT_PRESETS[self.variant].items():
            if getattr(self, name) is None:
                setattr(self, name, value)
        self.backbone = str(self.backbone)
        self.checkpoint = str(DEFAULT_MODEL_NAME if self.checkpoint is None else self.checkpoint)
        self.base_model = str(DEFAULT_BASE_MODEL if self.base_model is None else self.base_model)
        self.base_path = None if self.base_path is None else str(self.base_path)
        self.lora_rank = int(self.lora_rank)
        self.resolution = _as_resolution(self.resolution)
        self.num_frames = int(self.num_frames)
        self.fps = float(self.fps)
        self.block_index = int(self.block_index)
        self.max_steps = int(self.max_steps)
        self.num_steps = int(self.num_steps)
        self.cfg_scale = float(self.cfg_scale)
        self.sigma_shift = float(self.sigma_shift)
        self.flow_cond = _normalise_flow_cond(self.flow_cond)
        self.full_traj = _normalise_full_traj(self.full_traj)
        self.tia_inject = _normalise_switch(self.tia_inject, "tia_inject")
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
        if not math.isfinite(self.sigma_shift) or self.sigma_shift <= 0.0:
            raise ValueError(f"sigma_shift must be a positive shift, got {self.sigma_shift}")
        if self.lora_rank < 1:
            raise ValueError(f"lora_rank must be positive, got {self.lora_rank}")

    @classmethod
    def from_config(cls, cfg: Any, **overrides: Any) -> FlowWAMConfig:
        """Read the configuration out of a parsed run config.

        Args:
            cfg: Parsed YAML of a run, e.g. `configs/paper/flowwam/robotwin/eveworld.yaml`,
                as a mapping or a `DictConfig`. The `model` block carries the backbone and
                the geometry, `train` the LoRA rank and the step budget and `inference`
                the sampling protocol of the released checkpoint.
            **overrides: Fields that take precedence over the parsed values.

        Returns:
            The configuration; keys the run does not carry keep the preset or dataclass
            default, and the variant is inferred from the run name, the split file and the
            metadata directory when `model.variant` is absent.
        """
        values: dict[str, Any] = {
            "variant": _infer_variant(cfg),
            "backbone": cfg_get(cfg, "model.backbone", "flowwam"),
            "checkpoint": cfg_get(cfg, "model.checkpoint"),
            "base_model": cfg_get(cfg, "model.base"),
            "base_path": cfg_get(cfg, "model.base_path"),
            "lora_rank": cfg_get(cfg, "train.lora_rank"),
            "resolution": cfg_get(cfg, "model.resolution"),
            "num_frames": cfg_get(cfg, "model.num_frames"),
            "fps": cfg_get(cfg, "model.fps"),
            "block_index": cfg_get(cfg, "model.block_index"),
            "max_steps": cfg_get(cfg, "train.max_steps"),
            "num_steps": cfg_get(cfg, "inference.num_steps"),
            "cfg_scale": cfg_get(cfg, "inference.cfg_scale"),
            "sigma_shift": cfg_get(cfg, "inference.sigma_shift"),
            "flow_cond": cfg_get(cfg, "inference.flow_cond"),
            "tia_inject": cfg_get(cfg, "inference.tia_inject"),
            "full_traj": cfg_get(cfg, "inference.full_traj"),
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
        """Temporal size of the video latent."""
        return latent_frames(self.num_frames)

    @property
    def grid(self) -> tuple[int, int, int]:
        """`(T, h, w)` of the latent grid, i.e. of the IGR weight map before the resize."""
        return latent_grid(self)

    @property
    def latent_shape(self) -> tuple[int, int, int, int]:
        """`(C, T, h, w)` of the video latent the VAE produces."""
        return latent_shape(self)

    @property
    def tia_grid(self) -> tuple[int, int]:
        """`(rows, columns)` of the token grid of one latent frame, `(15, 20)` for RoboTwin."""
        return (
            self.height // (CELL_SIZE * PATCH_SIZE),
            self.width // (CELL_SIZE * PATCH_SIZE),
        )


def latent_frames(num_frames: int, stride: int = TEMPORAL_STRIDE) -> int:
    """Temporal size of the video latent for a clip of `num_frames` frames.

    Args:
        num_frames: Frames of the clip.
        stride: Frames covered by one latent frame, 4 for the released VAE.

    Returns:
        `1 + (num_frames - 1) // stride`, e.g. 8 for the 29-frame protocol clip and 31
        for a 121-frame window.
    """
    frames = int(num_frames)
    if frames < 1:
        raise ValueError(f"num_frames must be positive, got {num_frames}")
    return 1 + (frames - 1) // int(stride)


def latent_grid(config: Any = None) -> tuple[int, int, int]:
    """Latent grid of a clip, `(T, h, w)`, the grid the IGR weight map is resized from.

    Args:
        config: A :class:`FlowWAMConfig`, a mapping of its fields or a parsed run config;
            `None` uses the RoboTwin preset.

    Returns:
        `(1 + (num_frames - 1) // 4, height // 16, width // 16)`, which is `(8, 30, 40)`
        for the 29-frame `640x480` protocol clip and `(31, 30, 40)` for 121 frames.
    """
    cfg = _as_config(config)
    return (latent_frames(cfg.num_frames), cfg.height // CELL_SIZE, cfg.width // CELL_SIZE)


def latent_shape(config: Any = None) -> tuple[int, int, int, int]:
    """Shape of the video latent, `(C, T, h, w)`.

    Args:
        config: A :class:`FlowWAMConfig`, a mapping of its fields or a parsed run config;
            `None` uses the RoboTwin preset.

    Returns:
        `(48, 1 + (num_frames - 1) // 4, height // 16, width // 16)`, which is
        `(48, 8, 30, 40)` for the 29-frame `640x480` protocol clip.
    """
    cfg = _as_config(config)
    return (
        LATENT_CHANNELS,
        latent_frames(cfg.num_frames),
        cfg.height // VAE_SPATIAL_STRIDE,
        cfg.width // VAE_SPATIAL_STRIDE,
    )


def load_backbone(config: Any = None, *, device: Any = None, **overrides: Any) -> Any:
    """Build the released FlowWAM pipeline: DiT, flow stream, VAE and prompt encoder.

    The release ships its pipeline as a `WanVideoPipeline` with a `FlowStreamModule`
    beside the DiT, driven together by `diffsynth.pipelines.wan_video_dual_stream`. The
    loader of the release, `inference/pipeline_loader.build_pipeline`, is preferred
    because it applies the checkpoint and the fp32 modulation of the trained layers the
    way the release does; when the checkout is importable but that loader is not, the
    pipeline is built here from `diffsynth` directly. Either way the flow module is left
    on `pipeline.flow_stream` and a TIA adapter carried by the checkpoint is registered
    on the denoiser.

    Args:
        config: A :class:`FlowWAMConfig`, a mapping of its fields or a parsed run config;
            `None` uses the RoboTwin preset.
        device: Device the pipeline is built on; the visible CUDA device by default.
        **overrides: Configuration fields that take precedence over `config`.

    Returns:
        The pipeline, ready for :class:`FlowWAMModel`.

    Raises:
        ImportError: If neither the release checkout nor `diffsynth` is importable.
        FileNotFoundError: If the base checkpoint or the fine-tuned weights cannot be
            found under `FLOWWAM_MODEL_DIR`, `EVEWORLD_CHECKPOINT_ROOT` or `checkpoints/`.
    """
    cfg = _as_config(config, **overrides)
    searched = _ensure_importable()
    if not _module_available(_BACKBONE_MODULE) and not _module_available(_LOADER_MODULE):
        raise ImportError(
            f"FlowWAM needs the release checkout on `sys.path`; searched {searched}. Clone "
            f"it with scripts/setup/clone_flowwam.sh or point FLOWWAM_ROOT at an existing "
            f"clone."
        )
    target = _default_device() if device is None else torch.device(device)
    base_dir = _resolve_base(cfg)
    weight = _resolve_weight(cfg)
    logger.info("Loading FlowWAM weights %s over the base checkpoint %s", weight, base_dir)
    builder = _import_callable(_LOADER_MODULE, "build_pipeline")
    if builder is None:
        logger.info("%s is not importable, building the pipeline with diffsynth directly", _LOADER_MODULE)
        pipeline = _build_pipeline_fallback(cfg, target, base_dir, weight)
    else:
        pipeline, flow_stream = builder(local_model_path=str(base_dir), device=target, full_path=str(weight))
        _attach_flow_stream(pipeline, flow_stream)
    _register_checkpoint_adapter(pipeline, weight, cfg)
    return pipeline


_COMPONENT_NAMES: dict[str, tuple[str, ...]] = {
    "denoiser": ("dit", "denoiser", "transformer", "model"),
    "flow_stream": ("flow_stream", "flowstream", "flow_module"),
    "vae": ("vae", "video_vae"),
    "text_encoder": ("text_encoder", "text_encoder_2", "encoder"),
    "scheduler": ("scheduler", "noise_scheduler"),
    "tokenizer": ("tokenizer", "text_tokenizer"),
    "prompter": ("prompter", "prompt_encoder"),
}


class FlowWAMModel:
    """The FlowWAM backbone as the trainer and the evaluation scripts use it.

    The wrapper holds the released pipeline and exposes the pieces the rest of EVEWorld
    talks to: the denoiser, the flow stream, the VAE, the text encoder and the prompt
    encoder, plus :meth:`encode_text`, :meth:`encode_video` and :meth:`decode`. On top of
    those it carries the two halves of the training objective, :meth:`forward` and
    :meth:`igr_loss`, and :meth:`inference`, the sampling loop of the release's
    `inference/arm_generate.py`.

    Components are looked up on the pipeline under the spellings the released code uses,
    so a pipeline that renames its denoiser still works, and a component that is missing
    raises a `RuntimeError` naming what the pipeline does carry instead of failing with
    an `AttributeError` somewhere inside a sampling loop.
    """

    def __init__(
        self,
        config: Any = None,
        pipeline: Any = None,
        *,
        device: Any = None,
        dit: Any = None,
        flow_stream: Any = None,
        vae: Any = None,
        text_encoder: Any = None,
        scheduler: Any = None,
        tokenizer: Any = None,
        prompter: Any = None,
    ) -> None:
        """Wrap a pipeline, building one with :func:`load_backbone` when none is given.

        Args:
            config: A :class:`FlowWAMConfig`, a mapping of its fields or a parsed run
                config; `None` uses the RoboTwin preset.
            pipeline: The released pipeline. When `None` it is built with
                :func:`load_backbone`, unless `dit` is given, which means the caller
                brings its own components.
            device: Device the wrapped modules run on; the device of the pipeline when
                not given.
            dit: Denoiser, looked up on `pipeline` when `None`.
            flow_stream: Optical-flow stream of the dual-stream DiT.
            vae: Video VAE.
            text_encoder: Text encoder.
            scheduler: Noise scheduler, used for its timestep convention.
            tokenizer: Tokenizer of the prompt encoder.
            prompter: Prompt encoder, the component that turns prompts into contexts.
        """
        self.config = _as_config(config)
        self._device = None if device is None else torch.device(device)
        self._explicit = {
            name: value
            for name, value in (
                ("dit", dit),
                ("flow_stream", flow_stream),
                ("vae", vae),
                ("text_encoder", text_encoder),
                ("scheduler", scheduler),
                ("tokenizer", tokenizer),
                ("prompter", prompter),
            )
            if value is not None
        }
        if pipeline is not None:
            self.pipeline = pipeline
        elif dit is not None:
            self.pipeline = None
        else:
            self.pipeline = load_backbone(self.config, device=self._device)

    def _component(self, name: str, *, required: bool = True) -> Any:
        """Component of the pipeline under one of the names the release uses."""
        aliases = _COMPONENT_NAMES[name]
        for alias in aliases:
            value = self._explicit.get(alias)
            if value is not None:
                return value
        pipeline = self.pipeline
        if pipeline is None:
            if not required:
                return None
            raise RuntimeError(
                f"No {name!r} available: the model was built from explicit components and "
                f"none of {list(aliases)} was among them; pass one of them to "
                f"FlowWAMModel(...) or build the pipeline with load_backbone()."
            )
        for alias in aliases:
            value = getattr(pipeline, alias, None)
            if value is not None:
                return value
        if not required:
            return None
        raise RuntimeError(f"The pipeline carries no {name!r}, tried {list(aliases)}; it carries " f"{_public_names(pipeline)}.")

    @property
    def denoiser(self) -> Any:
        """The DiT, the denoiser both streams run through."""
        return self._component("denoiser")

    @property
    def backbone(self) -> Any:
        """The DiT under the name the configuration uses for the backbone family."""
        return self.denoiser

    @property
    def flow_stream(self) -> Any:
        """The optical-flow stream of the dual-stream DiT."""
        return self._component("flow_stream")

    @property
    def vae(self) -> Any:
        """The video VAE."""
        return self._component("vae")

    @property
    def text_encoder(self) -> Any:
        """The text encoder."""
        return self._component("text_encoder")

    @property
    def tokenizer(self) -> Any:
        """The tokenizer paired with the text encoder."""
        return self._component("tokenizer")

    @property
    def scheduler(self) -> Any:
        """The noise scheduler of the pipeline."""
        return self._component("scheduler")

    @property
    def prompter(self) -> Any:
        """The prompt encoder of the pipeline."""
        return self._component("prompter")

    @property
    def device(self) -> torch.device:
        """Device the model runs on, taken from the pipeline when not given explicitly."""
        if self._device is not None:
            return self._device
        raw = getattr(self.pipeline, "device", None)
        if raw is not None:
            return torch.device(raw)
        for module in (self._component("denoiser", required=False), self.vae):
            if isinstance(module, torch.nn.Module):
                parameter = next(module.parameters(), None)
                if parameter is not None:
                    return parameter.device
        return torch.device("cpu")

    @property
    def dtype(self) -> torch.dtype:
        """Dtype the model runs in, taken from the pipeline when it declares one."""
        raw = getattr(self.pipeline, "torch_dtype", None)
        if isinstance(raw, torch.dtype):
            return raw
        for module in (self._component("denoiser", required=False), self.vae):
            if isinstance(module, torch.nn.Module):
                parameter = next(module.parameters(), None)
                if parameter is not None:
                    return parameter.dtype
        return torch.float32

    @property
    def grid(self) -> tuple[int, int, int]:
        """`(T, h, w)` of the latent grid of the configured clip."""
        return self.config.grid

    @property
    def latent_shape(self) -> tuple[int, int, int, int]:
        """`(C, T, h, w)` of the video latent of the configured clip."""
        return self.config.latent_shape

    @property
    def tia_grid(self) -> tuple[int, int]:
        """`(rows, columns)` of the token grid TIA reshapes block activations onto."""
        return self.config.tia_grid

    def _named_modules(self) -> list[tuple[str, torch.nn.Module]]:
        """The pipeline and every component that is a module, once each."""
        modules: list[tuple[str, torch.nn.Module]] = []
        seen: set[int] = set()
        candidates: list[tuple[str, Any]] = [("pipeline", self.pipeline)]
        candidates += [(name, self._component(name, required=False)) for name in _COMPONENT_NAMES]
        for name, candidate in candidates:
            if isinstance(candidate, torch.nn.Module) and id(candidate) not in seen:
                seen.add(id(candidate))
                modules.append((name, candidate))
        return modules

    def parameters(self) -> Iterator[torch.nn.Parameter]:
        """Parameters of every component, each module contributing once."""
        for _, module in self._named_modules():
            yield from module.parameters()

    def named_parameters(self) -> Iterator[tuple[str, torch.nn.Parameter]]:
        """Parameters of every component, keys prefixed with the component name."""
        for name, module in self._named_modules():
            for key, parameter in module.named_parameters():
                yield f"{name}.{key}", parameter

    def train(self, mode: bool = True) -> FlowWAMModel:
        """Put every component into training mode."""
        for _, module in self._named_modules():
            module.train(mode)
        return self

    def eval(self) -> FlowWAMModel:
        """Put every component into evaluation mode."""
        return self.train(False)

    def to(self, *args: Any, **kwargs: Any) -> FlowWAMModel:
        """Move and cast every component, as `torch.nn.Module.to` does.

        The device and the dtype the pipeline declares are updated as well, so that
        later calls generate noise and latents in the dtype the modules actually run in.
        """
        for _, module in self._named_modules():
            module.to(*args, **kwargs)
        device = kwargs.get("device")
        if device is None:
            for argument in args:
                if isinstance(argument, (str, torch.device)):
                    device = argument
                    break
        if device is not None:
            self._device = torch.device(device)
        dtype = kwargs.get("dtype")
        if dtype is None and len(args) > 1 and isinstance(args[1], torch.dtype):
            dtype = args[1]
        if isinstance(dtype, torch.dtype) and self.pipeline is not None:
            if hasattr(self.pipeline, "torch_dtype"):
                self.pipeline.torch_dtype = dtype
        return self

    def encode_text(self, texts: Any, *, positive: bool = True, device: Any = None) -> torch.Tensor:
        """Encode prompts into the context the denoiser conditions on.

        Args:
            texts: A prompt or a list of prompts.
            positive: Whether the prompts are the positive ones; the negative pass of
                classifier-free guidance encodes with `positive=False`.
            device: Device of the returned tensor; the device of the model by default.

        Returns:
            `(B, L, C)` text embeddings, one row per prompt.

        Raises:
            RuntimeError: If the pipeline carries neither a prompt encoder nor a
                tokenizer to pair with the text encoder.
        """
        prompts = _as_text_list(texts)
        target = self.device if device is None else torch.device(device)
        prompter = self._component("prompter", required=False)
        chunks: list[torch.Tensor] = []
        if prompter is not None and callable(getattr(prompter, "encode_prompt", None)):
            for prompt in prompts:
                encoded = prompter.encode_prompt(prompt, positive=positive, device=target)
                chunks.append(_text_tensor(encoded).to(device=target))
            return torch.cat(chunks, dim=0)
        tokenizer = self._component("tokenizer", required=False)
        encoder = self._component("text_encoder", required=False)
        if tokenizer is None or encoder is None:
            raise RuntimeError(
                "The pipeline carries no way to encode text: expected a `prompter` with " "`encode_prompt` or a `tokenizer` beside the text encoder."
            )
        for prompt in prompts:
            tokens = tokenizer(prompt, return_tensors="pt")
            if isinstance(tokens, Mapping):
                inputs = {key: value.to(target) for key, value in tokens.items()}
            else:
                inputs = tokens.to(target)
            output = _call_component(encoder, (inputs,), device=target)
            chunks.append(_text_tensor(output).to(device=target))
        return torch.cat(chunks, dim=0)

    def encode_video(self, video: Any, *, device: Any = None) -> torch.Tensor:
        """Encode frames into video latents with the VAE.

        Args:
            video: A clip of shape `(T, H, W, 3)` or `(T, 3, H, W)`, or a batch of clips;
                values in `[0, 1]` or `[0, 255]` are rescaled to `[-1, 1]`, values already
                in `[-1, 1]` are passed through.
            device: Device the VAE runs on; the device of the model by default.

        Returns:
            `(C, T, h, w)` for a single clip, `(B, C, T, h, w)` for a batch.
        """
        clips, batched = _as_video_batch(video)
        target = self.device if device is None else torch.device(device)
        vae = self.vae
        latents = [_encode_clip(vae, clip, device=target) for clip in clips]
        if batched or len(latents) > 1:
            return torch.stack(latents, dim=0)
        return latents[0]

    def decode(self, latents: Any, *, device: Any = None) -> torch.Tensor:
        """Decode video latents back into frames.

        Args:
            latents: A latent of shape `(C, T, h, w)` or a batch `(B, C, T, h, w)`.
            device: Device the VAE runs on; the device of the model by default.

        Returns:
            Frames in `[-1, 1]`, `(T, 3, H, W)` for a single latent and `(B, T, 3, H, W)`
            for a batch.

        Raises:
            ValueError: If the latents have neither 4 nor 5 dimensions.
        """
        tensor = latents if torch.is_tensor(latents) else torch.as_tensor(latents)
        target = self.device if device is None else torch.device(device)
        vae = self.vae
        if tensor.dim() == 4:
            return _decode_clip(vae, tensor.to(target), device=target)
        if tensor.dim() == 5:
            rows = [_decode_clip(vae, row.to(target), device=target) for row in tensor]
            return torch.stack(rows, dim=0)
        raise ValueError(f"Expected latents of 4 or 5 dimensions, got {tuple(tensor.shape)}")

    def sample_sigma(
        self,
        batch_size: int,
        *,
        device: Any = None,
        dtype: Any = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Draw one training sigma per sample from the schedule the release trains with.

        Args:
            batch_size: Number of sigmas.
            device: Device of the returned tensor; the device of the model by default.
            dtype: Dtype of the returned tensor; the dtype of the model by default.
            generator: Random source; the default generator when not given.

        Returns:
            `(B,)` sigmas of the shifted linear ladder the release samples its training
            timesteps from, `sigma * 1000` being the value the denoiser is told.
        """
        target = self.device if device is None else torch.device(device)
        kind = self.dtype if dtype is None else dtype
        ladder = _train_sigmas(self.config.sigma_shift)
        index = torch.randint(0, int(ladder.shape[0]), (int(batch_size),), generator=generator)
        return ladder[index].to(device=target, dtype=kind)

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Weights of the denoiser, the flow stream and the registered TIA adapter.

        The keys follow the layout of the released checkpoints: denoiser weights under
        `dit.`, flow-stream weights under `flow_stream.` and adapter weights under
        `tia_adapter.`, so a checkpoint written here loads back through
        :func:`load_backbone` and through the loader of the release.
        """
        state: dict[str, torch.Tensor] = {}
        for key, value in self.denoiser.state_dict().items():
            state[f"dit.{key}"] = value.detach()
        flow_stream = self._component("flow_stream", required=False)
        if isinstance(flow_stream, torch.nn.Module):
            for key, value in flow_stream.state_dict().items():
                state[f"flow_stream.{key}"] = value.detach()
        state.update(_hooks().tia_state_dict(self.denoiser))
        return state

    def load_state_dict(self, state: Mapping[str, Any], *, strict: bool = False) -> dict[str, Any]:
        """Load a checkpoint into the denoiser, the flow stream and the TIA adapter.

        Keys under `dit.` go to the denoiser, keys under `flow_stream.` to the flow stream
        and keys under `tia_adapter.` to a TIA adapter, which is created when the model
        does not carry one yet.

        Args:
            state: Mapping of tensor keys to tensors, as :meth:`state_dict` writes it.
            strict: Whether keys outside the three groups are an error.

        Returns:
            Per group a `{"missing": [...], "unexpected": [...]}` report, and the keys
            that belonged to no group under `"skipped"`.

        Raises:
            ValueError: If `strict` and a key belongs to no group.
        """
        groups: dict[str, dict[str, Any]] = {"dit": {}, "flow_stream": {}, "tia_adapter": {}}
        skipped: list[str] = []
        for key, value in dict(state).items():
            for prefix in ("dit.", "flow_stream.", "tia_adapter."):
                if key.startswith(prefix):
                    groups[prefix[:-1]][key[len(prefix) :]] = value
                    break
            else:
                skipped.append(key)
        if strict and skipped:
            raise ValueError(f"Unrecognised checkpoint keys: {sorted(skipped)}")
        report: dict[str, Any] = {"skipped": skipped}
        denoiser = self.denoiser
        missing, unexpected = denoiser.load_state_dict(groups["dit"], strict=False)
        report["dit"] = {"missing": list(missing), "unexpected": list(unexpected)}
        flow_stream = self._component("flow_stream", required=False)
        if groups["flow_stream"]:
            if not isinstance(flow_stream, torch.nn.Module):
                report["flow_stream"] = {"missing": [], "unexpected": sorted(groups["flow_stream"])}
                logger.warning("Checkpoint carries flow-stream weights the model cannot take")
            else:
                missing, unexpected = flow_stream.load_state_dict(groups["flow_stream"], strict=False)
                report["flow_stream"] = {"missing": list(missing), "unexpected": list(unexpected)}
        if groups["tia_adapter"]:
            adapter = _hooks().register_tia(
                denoiser,
                state_dict=groups["tia_adapter"],
                layer_index=self.config.block_index,
                grid=self.config.tia_grid,
            )
            report["tia_adapter"] = {
                "missing": [],
                "unexpected": [] if adapter is not None else sorted(groups["tia_adapter"]),
            }
        return report

    def forward(
        self,
        batch: Mapping[str, Any],
        *,
        context: Any = None,
        negative_context: Any = None,
        generator: torch.Generator | None = None,
        gradient_checkpointing: bool = False,
    ) -> dict[str, Any]:
        """One step of the dual-stream objective, without the loss.

        Both streams are noised with the same shifted-linear sigma the release trains
        with, the denoiser is asked for the velocity of each stream and the first latent
        frame of both streams is pinned to its clean value because it carries the
        conditioning frame. The RGB stream is noised around the *disturbed* clip and
        compared against the target velocity of the *clean* one, which is the
        restoration set-up of `eveworld/flowwam_port/eve_flowwam_train.py`; the loss
        itself lives in :meth:`igr_loss` and in the flow term of the trainer, so the two
        can be weighted outside this method.

        Args:
            batch: Sample or collated batch. `video` carries the demonstration clip and
                `corrected` (also `video_corrected` or `igr_video`) the disturbed clip
                the model restores, which defaults to the demonstration. A `flow` (also
                `flow_video`) clip conditions the flow stream; without one the stream is
                supervised on the blank clip instead, which is what a dataset without
                rendered robot video asks for.
            context: `(B, L, C)` text embeddings of the batch; encoded from its `caption`
                (`instruction`, `prompt`) when not given.
            negative_context: Accepted for the call surface of the trainer and unused,
                because classifier-free guidance only applies while sampling.
            generator: Random source of the noise and of the sigma draw.
            gradient_checkpointing: Trade compute for memory inside the denoiser.

        Returns:
            The velocity predictions and their targets, the clean-latent estimates, the
            IGR map of the batch, the sampled sigma and the noise. A batch that carries
            flow frames additionally gets `flow_pred`, `flow_target`, `flow_latents`,
            `flow_pred_velocity`, `flow_target_velocity` and `flow_noise`.

        Raises:
            KeyError: If the batch carries no `video` entry, or no caption to condition on
                while `context` is not given.
            RuntimeError: If `diffsynth` is not importable, or the dual-stream model
                function returns something other than a pair of velocities.
        """
        video = _first_value(batch, ("video", "frames", "rgb"))
        if video is None:
            raise KeyError("The batch carries no `video` entry to train the RGB stream on")
        target = _as_rank5(self.encode_video(video))
        source_video = _first_value(batch, ("corrected", "video_corrected", "igr_video"))
        source = target if source_video is None else _as_rank5(self.encode_video(source_video))
        flow_video = _first_value(batch, ("flow", "flow_video"))
        if flow_video is None:
            flow_latents = _as_rank5(self.encode_video(_blank_video(video)))
        else:
            flow_latents = _as_rank5(self.encode_video(flow_video))
        batch_size = int(target.shape[0])
        sigma = self.sample_sigma(batch_size, device=target.device, dtype=torch.float32, generator=generator)
        level = _batch_sigma(sigma).to(device=target.device, dtype=target.dtype)
        rgb_noise = _randn(target.shape, generator=generator, device=target.device, dtype=target.dtype)
        flow_noise = _randn(flow_latents.shape, generator=generator, device=target.device, dtype=target.dtype)
        rgb_noisy = _pin_first_frame((1.0 - level) * source + level * rgb_noise, source)
        flow_noisy = _pin_first_frame((1.0 - level) * flow_latents + level * flow_noise, flow_latents)
        if context is None:
            context = self.encode_text(_batch_captions(batch, batch_size), positive=True, device=target.device)
        timestep = sigma.reshape(-1) * _TIMESTEP_SCALE
        rgb_velocity, flow_velocity = _call_dual_stream(
            self,
            rgb_noisy,
            flow_noisy,
            timestep,
            context,
            gradient_checkpointing=gradient_checkpointing,
        )
        outputs: dict[str, Any] = {
            "pred": rgb_noisy - level * rgb_velocity,
            "target": target,
            "weight_map": _first_value(batch, ("weight_map", "tia_weight_map", "w_map")),
            "sigma": sigma,
            "noise": rgb_noise,
            "source": source,
            "timestep": timestep,
            "pred_velocity": rgb_velocity,
            "target_velocity": rgb_noise - target,
        }
        if flow_video is not None:
            outputs["flow_pred"] = flow_noisy - level * flow_velocity
            outputs["flow_target"] = flow_latents
            outputs["flow_latents"] = flow_latents
            outputs["flow_pred_velocity"] = flow_velocity
            outputs["flow_target_velocity"] = flow_noise - flow_latents
            outputs["flow_noise"] = flow_noise
        return outputs

    def igr_loss(
        self,
        pred: Any,
        target: Any,
        weight_map: Any = None,
        sigma: Any = None,
        *,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """The IGR restoration term of one step, on the latent grid.

        The release trains in velocity space: the network predicts `noise - clean` and
        the term compares that prediction against the target velocity under the IGR
        weight map, with the first latent frame masked because it is pinned to the
        conditioning frame. The map is put on the latent grid with
        :func:`eveworld.data.transforms.latent.resize_weight_map`, which keeps its unit
        mean, and is then normalised by the mean of the frames after the first, the
        normalisation `eveworld/flowwam_port/eve_flowwam_train.py` applies.

        Unlike :func:`eveworld.methods.igr.loss.igr_loss` this term carries no EDM
        weighting of the sampled level, because the released flow-matching run does not
        weight the term that way.

        Args:
            pred: Predicted velocity of the RGB stream, `(B, C, T, h, w)` or `(C, T, h, w)`.
            target: Target velocity, same shape as `pred`.
            weight_map: `(h, w)`, `(T, h, w)` or `(B, T, h, w)` IGR map of the batch, on the
                pixel grid of the annotations or on the latent grid. `None` weights every
                cell of every frame after the first equally.
            sigma: Sampled noise level, accepted for the call surface of the trainer and
                unused, because the velocity-space term is not rescaled with it.
            eps: Positive floor of the normalisation mean.

        Returns:
            Scalar loss tensor in the dtype of `pred`.

        Raises:
            ValueError: If `pred` and `target` differ in shape, either is not a rank-4 or
                rank-5 latent, `eps` is not positive, or the weight map does not broadcast
                against the latent.
        """
        if eps <= 0.0:
            raise ValueError(f"eps must be positive, got {eps}")
        prediction = _as_rank5(pred).float()
        truth = _as_rank5(target).float()
        if prediction.shape != truth.shape:
            raise ValueError(f"pred and target must share a shape, got {tuple(prediction.shape)} and " f"{tuple(truth.shape)}")
        frames = int(prediction.shape[2])
        mask = torch.ones(
            (prediction.shape[0], 1, frames, prediction.shape[3], prediction.shape[4]),
            dtype=prediction.dtype,
            device=prediction.device,
        )
        if frames > 1:
            mask[:, :, :1] = 0.0
        weight = mask
        if weight_map is not None:
            weight = weight * _weight_map_for(
                weight_map,
                (prediction.shape[3], prediction.shape[4]),
                num_frames=frames,
                device=prediction.device,
                dtype=prediction.dtype,
                eps=eps,
            )
        difference = prediction - truth
        numerator = (weight * difference * difference).sum()
        denominator = weight.expand_as(difference).sum().clamp(min=eps)
        return numerator / denominator

    def inference(
        self,
        prompt: str = "",
        *,
        negative_prompt: str | None = None,
        image: Any = None,
        flow_video: Any = None,
        num_frames: int | None = None,
        num_steps: int | None = None,
        cfg_scale: float | None = None,
        seed: int | None = None,
        fps: float | None = None,
        flow_cond: Any = None,
        tia_inject: Any = None,
        full_traj: Any = None,
    ) -> np.ndarray:
        """Generate a clip with the sampling protocol of the release's `arm_generate.py`.

        Args:
            prompt: Text prompt of the scene.
            negative_prompt: Prompt of the negative pass of classifier-free guidance, the
                empty prompt when not given. The pass is skipped at `cfg_scale == 1.0`.
            image: First frame the clip is conditioned on, a PIL image, an array or a
                tensor. `None` starts from a white frame, the fallback of the released
                inference script when it is not given a render.
            flow_video: Robot-only frames, a clip of shape `(T, H, W, 3)` or the path of a
                render to read, that the flow stream is conditioned on when `flow_cond` is
                `"robot_only"`. `None` samples the flow stream from noise.
            num_frames: Frames of the clip; the frame count of the configuration by
                default.
            num_steps: Sampling steps; the step count of the configuration by default.
            cfg_scale: Classifier-free guidance scale; the scale of the configuration by
                default.
            seed: Seed of the noise draw; :data:`_DEFAULT_SEED` by default.
            fps: Frame rate of the clip, accepted for the call surface of the evaluation
                harness and unused: the backbone is conditioned on the frame rate of the
                training data, not on a value passed at sampling time.
            flow_cond: `"robot_only"` or `"none"`; the configuration by default.
            tia_inject: Whether the registered TIA adapter is injected into the pass; the
                configuration by default.
            full_traj: `"direct"` samples the clip in one pass, `"on"` rolls 121-frame
                windows with cross-faded overlaps; the configuration by default.

        Returns:
            Frames as a `(T, H, W, 3)` `uint8` array.

        Raises:
            ValueError: If the frame count is below two, the step count is below one, or a
                sampling option is not one of the accepted spellings.
            RuntimeError: If the pipeline carries no way to encode text, or the sampler
                cannot read the velocity of both streams out of the dual-stream pass.
        """
        config = self.config
        frames = int(config.num_frames if num_frames is None else num_frames)
        if frames < 2:
            raise ValueError(f"num_frames must be at least 2, got {frames}")
        steps = int(config.num_steps if num_steps is None else num_steps)
        if steps < 1:
            raise ValueError(f"num_steps must be at least 1, got {steps}")
        guidance = float(config.cfg_scale if cfg_scale is None else cfg_scale)
        base_seed = _DEFAULT_SEED if seed is None else int(seed)
        mode = config.flow_cond if flow_cond is None else _normalise_flow_cond(flow_cond)
        inject = config.tia_inject if tia_inject is None else _normalise_switch(tia_inject, "tia_inject")
        rollout = config.full_traj if full_traj is None else _normalise_full_traj(full_traj)
        context = self.encode_text(prompt, positive=True)
        negative: torch.Tensor | None = None
        if guidance != 1.0:
            text = "" if negative_prompt is None else negative_prompt
            negative = self.encode_text(text, positive=False)
        with _hooks().tia_injection(self.denoiser, inject):
            if rollout == "on":
                return self._roll_windows(
                    context,
                    negative,
                    num_frames=frames,
                    num_steps=steps,
                    cfg_scale=guidance,
                    seed=base_seed,
                    flow_cond=mode,
                    image=image,
                    flow_video=flow_video,
                )
            latent = self._sample_clip(
                context,
                negative,
                num_frames=frames,
                num_steps=steps,
                cfg_scale=guidance,
                seed=base_seed,
                flow_cond=mode,
                image=image,
                flow_video=flow_video,
            )
        return _as_frames(self.decode(latent)[0])

    def _sample_clip(
        self,
        context: torch.Tensor,
        negative_context: torch.Tensor | None,
        *,
        num_frames: int,
        num_steps: int,
        cfg_scale: float,
        seed: int,
        flow_cond: str,
        image: Any = None,
        flow_video: Any = None,
        start: int = 0,
    ) -> torch.Tensor:
        """Sample one clip from noise with the Euler loop of the released script.

        The first latent frame of both streams stays pinned to its condition after every
        step, and the flow stream is only stepped when it is not conditioned on a rendered
        robot-only video.

        Args:
            context: `(B, L, C)` positive text context.
            negative_context: `(B, L, C)` context of the negative pass, `None` to skip it.
            num_frames: Frames of the clip.
            num_steps: Euler steps.
            cfg_scale: Classifier-free guidance scale.
            seed: Seed of the noise draw of the RGB stream; the flow stream draws from
                `seed + 1`.
            flow_cond: `"robot_only"` conditions on `flow_video`, `"none"` samples the
                flow stream from noise.
            image: First frame of the clip, `None` for a white frame.
            flow_video: Robot-only render that conditions the flow stream.
            start: Index of the clip in the trajectory, the offset the window reads its
                flow video from.

        Returns:
            Latent of the clip, `(1, C, T, h, w)`.

        Raises:
            RuntimeError: If the dual-stream pass does not return the velocity of both
                streams.
        """
        device = self.device
        dtype = self.dtype
        config = self.config
        vae = self.vae
        scale = _vae_scale(vae)
        height, width = _frame_size(self.pipeline, config.height, config.width, num_frames)
        shape = (
            1,
            _latent_channels(vae),
            latent_frames(num_frames),
            height // scale,
            width // scale,
        )
        generator = torch.Generator("cpu").manual_seed(int(seed))
        rgb = _randn(shape, generator=generator, device=device, dtype=dtype)
        if image is None:
            logger.warning("No conditioning frame was given: the clip starts from a white frame")
            image = _white_frame(width, height)
        rgb[:, :, :1] = _prefix_latents(vae, image, height=height, width=width, device=device, dtype=dtype)
        conditioned = flow_cond == "robot_only" and flow_video is not None
        if conditioned:
            flow = _flow_condition_latents(
                flow_video,
                vae=vae,
                num_frames=num_frames,
                height=height,
                width=width,
                device=device,
                dtype=dtype,
                start=int(start),
            )
        else:
            generator = torch.Generator("cpu").manual_seed(int(seed) + 1)
            flow = _randn(shape, generator=generator, device=device, dtype=dtype)
            flow[:, :, :1] = _prefix_latents(
                vae,
                _white_frame(width, height),
                height=height,
                width=width,
                device=device,
                dtype=dtype,
            )
        sigmas, _ = _flow_schedule(num_steps, config.sigma_shift)
        sigmas = sigmas.to(device=device, dtype=torch.float32)
        prefix_rgb = rgb[:, :, :1].clone()
        prefix_flow = flow[:, :, :1].clone()
        with torch.no_grad():
            for index in range(num_steps):
                sigma = sigmas[index]
                step = (sigmas[index + 1] - sigma).to(dtype=rgb.dtype)
                timestep = sigma.reshape(1) * _TIMESTEP_SCALE
                rgb_velocity, flow_velocity = _call_dual_stream(self, rgb, flow, timestep, context, flow_cond=conditioned)
                if cfg_scale != 1.0 and negative_context is not None:
                    negative_rgb, negative_flow = _call_dual_stream(self, rgb, flow, timestep, negative_context, flow_cond=conditioned)
                    rgb_velocity = negative_rgb + cfg_scale * (rgb_velocity - negative_rgb)
                    flow_velocity = negative_flow + cfg_scale * (flow_velocity - negative_flow)
                rgb = rgb + step * rgb_velocity.to(dtype=rgb.dtype)
                rgb[:, :, :1] = prefix_rgb
                if not conditioned:
                    flow = flow + step * flow_velocity.to(dtype=flow.dtype)
                    flow[:, :, :1] = prefix_flow
        return rgb

    def _roll_windows(
        self,
        context: torch.Tensor,
        negative_context: torch.Tensor | None,
        *,
        num_frames: int,
        num_steps: int,
        cfg_scale: float,
        seed: int,
        flow_cond: str,
        image: Any = None,
        flow_video: Any = None,
    ) -> np.ndarray:
        """Generate a long trajectory by rolling 121-frame windows with cross-fades.

        The windows of the released script overlap by eleven frames: the first ten frames
        of a window are blended with the tail of its predecessor and the eleventh is
        dropped, which keeps the trajectory continuous without a visible cut. The first
        frame of every window after the first is the frame its predecessor produced at the
        start of that overlap, before the blend.

        Args:
            context: `(B, L, C)` positive text context.
            negative_context: `(B, L, C)` context of the negative pass, `None` to skip it.
            num_frames: Frames of the whole trajectory.
            num_steps: Euler steps per window.
            cfg_scale: Classifier-free guidance scale.
            seed: Seed of the first window; window `k` draws from `seed + 1000 * k`.
            flow_cond: `"robot_only"` or `"none"`.
            image: First frame of the trajectory, `None` for a white frame.
            flow_video: Robot-only render of the whole trajectory.

        Returns:
            Frames as a `(T, H, W, 3)` `uint8` array.

        Raises:
            ValueError: If a window would overlap its predecessor by less than one frame,
                or the windows come out shorter than `num_frames` frames in total.
        """
        starts = _window_starts(num_frames)
        clips: list[np.ndarray] = []
        for index, start in enumerate(starts):
            length = min(_WINDOW_FRAMES, num_frames - start)
            first = image if index == 0 else clips[index - 1][start - starts[index - 1]]
            latent = self._sample_clip(
                context,
                negative_context,
                num_frames=length,
                num_steps=num_steps,
                cfg_scale=cfg_scale,
                seed=int(seed) + 1000 * index,
                flow_cond=flow_cond,
                image=first,
                flow_video=flow_video,
                start=int(start),
            )
            clips.append(_as_frames(self.decode(latent)[0]))
        return _blend_windows(clips, starts, num_frames)


def _normalise_name(name: Any) -> str:
    """Fold a checkpoint name to the spelling the alias tables are keyed on."""
    text = str(name).strip().lower()
    return "".join(character if character.isalnum() else "_" for character in text)


def _checkpoint_roots() -> list[Path]:
    """Directories a released checkpoint is looked up under, in order."""
    roots: list[Path] = []
    for variable in ("FLOWWAM_MODEL_DIR", "EVEWORLD_CHECKPOINT_ROOT"):
        value = os.environ.get(variable)
        if value:
            roots.append(Path(value).expanduser())
    roots.append(repo_root() / "checkpoints")
    return roots


def _ensure_importable() -> list[str]:
    """Put the release checkout and its `inference/` directory on `sys.path`.

    `inference/pipeline_loader.py` is a top-level module of the release while `diffsynth`
    sits at the root of the checkout, so both directories are needed. The directories are
    appended rather than prepended, so an installed `diffsynth` keeps precedence over the
    copy that ships with a clone.

    Returns:
        Every path that was tried, in order, for the error message of :func:`load_backbone`.
    """
    roots: list[Path] = []
    for variable in ("EVEWORLD_FLOWWAM_ROOT", "FLOWWAM_ROOT"):
        value = os.environ.get(variable)
        if value:
            roots.append(Path(value).expanduser())
    roots.append(repo_root() / _THIRD_PARTY_CHECKOUT)
    roots.append(Path.home() / "FlowWAM")
    tried: list[str] = []
    for root in roots:
        for directory in (root, root / "inference"):
            text = str(directory)
            if text not in tried:
                tried.append(text)
            if directory.is_dir() and text not in sys.path:
                sys.path.append(text)
    importlib.invalidate_caches()
    return tried


def _module_available(name: str) -> bool:
    """Whether a module can be imported, without importing it."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _import_callable(module: str, attribute: str) -> Any | None:
    """The named callable of a module, or `None` when the module does not carry it."""
    try:
        imported = importlib.import_module(module)
    except ImportError:
        return None
    candidate = getattr(imported, attribute, None)
    return candidate if callable(candidate) else None


def _default_device() -> torch.device:
    """The visible CUDA device, or the CPU when there is none."""
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _resolve_base(cfg: FlowWAMConfig) -> Path:
    """Locate the directory of the Wan base checkpoint the release is built on.

    Args:
        cfg: Configuration whose `base_path` names a directory directly, or whose
            `base_model` is resolved under :func:`_checkpoint_roots`.

    Returns:
        The resolved directory.

    Raises:
        FileNotFoundError: If no candidate directory exists, listing every path that was
            tried.
    """
    if cfg.base_path:
        explicit = Path(str(cfg.base_path)).expanduser()
        if explicit.is_dir():
            return explicit.resolve()
        raise FileNotFoundError(
            f"model.base_path {cfg.base_path!r} is not a directory; point it at a Wan base " "checkpoint directory such as checkpoints/wan2.2-ti2v-5b"
        )
    value = str(cfg.base_model).strip()
    normalised = _normalise_name(value)
    name = _BASE_ALIASES.get(normalised, normalised)
    tried: list[Path] = []
    for candidate in (Path(value).expanduser(), Path(name).expanduser()):
        if candidate not in tried:
            tried.append(candidate)
        for root in _checkpoint_roots():
            for path in (root / name, root / Path(name).name):
                if path not in tried:
                    tried.append(path)
    for candidate in tried:
        if candidate.is_dir():
            return candidate.resolve()
    listing = "\n".join(f"  - {path}" for path in tried)
    raise FileNotFoundError(
        f"Wan base checkpoint {value!r} was not found; tried:\n{listing}\n"
        "download it with scripts/setup/download_models.sh (which honours "
        "EVEWORLD_CHECKPOINT_ROOT), point FLOWWAM_MODEL_DIR at it, or pass model.base_path; "
        "see docs/checkpoints.md"
    )


def _resolve_weight(cfg: FlowWAMConfig) -> Path:
    """Locate the file holding the fine-tuned DiT, flow-stream and adapter weights.

    A `model.checkpoint` that names a file or a directory is used as given, otherwise the
    value is folded through :data:`_WEIGHT_ALIASES`, which maps the released checkpoint
    names onto the layout of `checkpoints/`, and the resulting names are looked up under
    :func:`_checkpoint_roots` and under `flowwam/` inside them.

    Args:
        cfg: Configuration whose `checkpoint` names the released weights.

    Returns:
        The resolved `.safetensors` file.

    Raises:
        FileNotFoundError: If no candidate exists, or a candidate directory holds more
            than one checkpoint, listing every path that was tried.
    """
    value = "" if cfg.checkpoint is None else str(cfg.checkpoint).strip()
    if not value:
        raise FileNotFoundError(
            "model.checkpoint is empty; name the released weights such as " f"{DEFAULT_MODEL_NAME} or point it at a .safetensors file"
        )
    normalised = _normalise_name(value)
    alias = _WEIGHT_ALIASES.get(normalised)
    names = [name for name in (alias, normalised) if name]
    if normalised:
        names.extend([f"{normalised}.safetensors", f"flowwam/{normalised}.safetensors"])
    roots = _checkpoint_roots()
    tried: list[Path] = [Path(value).expanduser()]
    for name in names:
        name_path = Path(name)
        tried.append(name_path.expanduser())
        for root in roots:
            tried.append(root / name)
            if name_path.parent != Path("."):
                tried.append(root / name_path.name)
    unique: list[Path] = []
    for candidate in tried:
        if candidate not in unique:
            unique.append(candidate)
    tried = unique
    for candidate in tried:
        if candidate.is_file():
            return candidate.resolve()
        if candidate.is_dir():
            files = sorted(path for path in candidate.glob("*.safetensors") if path.is_file())
            if len(files) == 1:
                return files[0].resolve()
            if len(files) > 1:
                listing = "\n".join(f"  - {path}" for path in files)
                raise FileNotFoundError(
                    f"model.checkpoint {value!r} names the directory {candidate}, which holds "
                    f"{len(files)} checkpoints; point it at one of:\n{listing}"
                )
    listing = "\n".join(f"  - {path}" for path in tried)
    raise FileNotFoundError(
        f"FlowWAM weights {value!r} were not found; tried:\n{listing}\n"
        "fetch them with scripts/setup/clone_flowwam.sh and "
        "scripts/setup/download_models.sh (which honour EVEWORLD_CHECKPOINT_ROOT), point "
        "FLOWWAM_MODEL_DIR at the directory holding them, or pass model.checkpoint; see "
        "docs/checkpoints.md"
    )


def _load_safetensors(path: Any) -> dict[str, torch.Tensor]:
    """Read a checkpoint into a plain mapping of tensors, without touching the GPU.

    A path that does not exist loads as an empty mapping with a warning, so callers that
    treat the weights as optional do not have to check twice.
    """
    target = Path(path).expanduser()
    if not target.is_file():
        logger.warning("Checkpoint %s does not exist, treating it as empty", target)
        return {}
    try:
        from safetensors.torch import load_file  # noqa: PLC0415 - optional dependency
    except ImportError:
        load_file = None
    if load_file is not None:
        try:
            return {str(key): value for key, value in dict(load_file(str(target))).items()}
        except (OSError, RuntimeError, ValueError) as error:
            logger.warning("Reading %s with safetensors failed (%s), falling back to torch", target, error)
    state = torch.load(str(target), map_location="cpu", weights_only=True)
    return {str(key): value for key, value in dict(state).items()}


def _attach_flow_stream(pipeline: Any, flow_stream: Any) -> None:
    """Leave the flow stream of the release on the pipeline, where the model looks for it."""
    if flow_stream is None:
        logger.warning("The loader returned no flow stream, the flow branch will be inactive")
        return
    existing = getattr(pipeline, "flow_stream", None)
    if existing is not None and existing is not flow_stream:
        logger.warning("Replacing the flow stream the pipeline already carried")
    setattr(pipeline, "flow_stream", flow_stream)


def _register_checkpoint_adapter(pipeline: Any, weight: Any, cfg: FlowWAMConfig) -> Any:
    """Register the TIA adapter a checkpoint carries on the denoiser of the pipeline.

    Args:
        pipeline: The pipeline whose denoiser takes the adapter.
        weight: The fine-tuned checkpoint to read the `tia_adapter.` keys from.
        cfg: Configuration naming the block the adapter belongs to and its token grid.

    Returns:
        The registered adapter, or `None` when the checkpoint carries none or the hooks
        module could not build one.
    """
    dit = getattr(pipeline, "dit", None)
    if dit is None:
        dit = getattr(pipeline, "denoiser", None)
    if dit is None:
        logger.warning(
            "The pipeline carries no denoiser to register a TIA adapter on; it holds %s",
            _public_names(pipeline),
        )
        return None
    state = {key[len("tia_adapter.") :]: value for key, value in _load_safetensors(weight).items() if key.startswith("tia_adapter.")}
    if not state:
        return None
    try:
        adapter = _hooks().register_tia(dit, state_dict=state, layer_index=cfg.block_index, grid=cfg.tia_grid)
    except (KeyError, RuntimeError, TypeError, ValueError) as error:
        logger.warning("Could not register the TIA adapter carried by %s: %s", weight, error)
        return None
    logger.info(
        "Registered the TIA adapter carried by %s on block %d with a %dx%d token grid",
        weight,
        cfg.block_index,
        cfg.tia_grid[0],
        cfg.tia_grid[1],
    )
    return adapter


def _restore_fp32_modulation(dit: Any, values: Mapping[str, torch.Tensor]) -> int:
    """Restore the fp32 precision of the trained modulation, time-MLP and LayerNorm weights.

    The release trains those weights in fp32 and casts them back for every forward pass,
    so loading the checkpoint in bfloat16 alone loses the precision the release runs with.
    The behaviour mirrors `_apply_fp32_modulation` of `inference/pipeline_loader.py`.

    Args:
        dit: The denoiser to restore the weights on.
        values: The fp32 entries of the checkpoint, keyed by parameter name.

    Returns:
        The number of parameters that were restored.

    Raises:
        ImportError: If the VRAM-management layers of `diffsynth` are not importable, which
            is the only place the release's cast wrappers are defined.
    """
    from diffsynth.vram_management.layers import AutoWrappedLinear, WanAutoCastLayerNorm

    parameters = dict(dit.named_parameters())
    restored = 0
    for key, value in values.items():
        parameter = parameters.get(key)
        if parameter is None:
            continue
        parameter.data = value.to(device=parameter.device)
        restored += int(parameter.numel())
    sequences = [module for module in (getattr(dit, "time_embedding", None), getattr(dit, "time_projection", None)) if module is not None]
    for sequence in sequences:
        for module in sequence.modules():
            if isinstance(module, AutoWrappedLinear):
                module.offload_dtype = torch.float32
                module.onload_dtype = torch.float32
                module.computation_dtype = torch.float32

    def _to_fp32(_module: Any, args: tuple[Any, ...]) -> tuple[Any, ...]:
        return tuple(item.float() if isinstance(item, torch.Tensor) else item for item in args)

    def _to_bfloat16(_module: Any, _args: Any, output: Any) -> Any:
        return output.bfloat16() if isinstance(output, torch.Tensor) else output

    for sequence in sequences:
        sequence.register_forward_pre_hook(_to_fp32)
        sequence.register_forward_hook(_to_bfloat16)
    for module in dit.modules():
        if isinstance(module, WanAutoCastLayerNorm):
            module.offload_dtype = torch.float32
            module.onload_dtype = torch.float32
    return restored


def _build_pipeline_fallback(cfg: FlowWAMConfig, device: torch.device, base_dir: Path, weight: Path) -> Any:
    """Build the released pipeline out of `diffsynth` when its own loader is missing.

    Mirrors `inference/pipeline_loader.build_pipeline`: the Wan2.2 text encoder, DiT and
    VAE are read from `base_dir`, the fine-tuned weights are split into their denoiser and
    flow-stream halves, the fp32 modulation of the trained checkpoint is restored and the
    flow stream is attached to the pipeline.

    Args:
        cfg: Configuration of the run; the LoRA rank and the block index are not used,
            because the fine-tuned checkpoint already carries the trained weights.
        device: Device the pipeline is built on.
        base_dir: Directory of the Wan base checkpoint.
        weight: Fine-tuned checkpoint to load over the base weights.

    Returns:
        The pipeline, with the flow stream attached.

    Raises:
        RuntimeError: If `diffsynth` is not importable, or the pipeline cannot be built.
    """
    try:
        from diffsynth.models.wan_video_dit_dual_stream import init_flow_stream  # noqa: PLC0415
        from diffsynth.pipelines.wan_video_new import ModelConfig, WanVideoPipeline  # noqa: PLC0415
    except ImportError as error:
        raise RuntimeError(
            f"diffsynth is not importable ({error}); install it with "
            "scripts/setup/clone_flowwam.sh and put the checkout on PYTHONPATH, or point "
            "EVEWORLD_FLOWWAM_ROOT at a clone"
        ) from error
    patterns = (
        "models_t5_umt5-xxl-enc-bf16.pth",
        "diffusion_pytorch_model*.safetensors",
        "Wan2.2_VAE.pth",
    )
    try:
        pipeline = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device=str(device),
            model_configs=[
                ModelConfig(
                    model_id="Wan-AI/Wan2.2-TI2V-5B",
                    origin_file_pattern=pattern,
                    offload_device="cpu",
                    local_model_path=str(base_dir),
                )
                for pattern in patterns
            ],
            tokenizer_config=ModelConfig(
                model_id="Wan-AI/Wan2.1-T2V-1.3B",
                origin_file_pattern="google/*",
                local_model_path=str(base_dir),
            ),
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise RuntimeError(f"Building the Wan2.2 pipeline from {base_dir} failed: {error}") from error
    flow_stream = init_flow_stream(pipeline.dit)
    dit_keys: dict[str, torch.Tensor] = {}
    flow_keys: dict[str, torch.Tensor] = {}
    for key, value in _load_safetensors(weight).items():
        if key.startswith("action_expert."):
            continue
        if key.startswith("flow_stream."):
            flow_keys[key[len("flow_stream.") :]] = value
        else:
            dit_keys[key] = value
    fp32_values = {key: value.clone() for key, value in dit_keys.items() if value.dtype == torch.float32}
    if dit_keys:
        missing, unexpected = pipeline.dit.load_state_dict(dit_keys, strict=False)
        logger.info(
            "Loaded %d denoiser weights from %s, %d missing and %d unexpected",
            len(dit_keys) - len(unexpected),
            weight,
            len(missing),
            len(unexpected),
        )
    if flow_keys:
        missing, unexpected = flow_stream.load_state_dict(flow_keys, strict=False)
        logger.info(
            "Loaded %d flow-stream weights from %s, %d missing and %d unexpected",
            len(flow_keys) - len(unexpected),
            weight,
            len(missing),
            len(unexpected),
        )
    if device.type == "cuda":
        pipeline.enable_vram_management()
    else:
        logger.info(
            "Running the pipeline on %s without the VRAM management of the release, which " "needs a CUDA device",
            device,
        )
    if fp32_values:
        try:
            restored = _restore_fp32_modulation(pipeline.dit, fp32_values)
        except ImportError as error:
            logger.warning("Could not restore the fp32 modulation of the checkpoint: %s", error)
        else:
            logger.info("Restored %d fp32 parameters of the trained layers", restored)
    flow_stream = flow_stream.to(device=device, dtype=torch.bfloat16).eval()
    _attach_flow_stream(pipeline, flow_stream)
    logger.info("Built the FlowWAM pipeline for the %s variant on %s", cfg.variant, device)
    return pipeline


def _signature_names(function: Any) -> frozenset[str] | None:
    """Names of the parameters a callable accepts, `None` when it cannot be inspected.

    A signature that ends in `**kwargs` carries `"**kwargs"`, which lets the callers below
    tell an explicit parameter from a catch-all.
    """
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return None
    names: set[str] = set()
    for parameter in signature.parameters.values():
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            names.add("**kwargs")
        else:
            names.add(parameter.name)
    return frozenset(names)


def _call_component(
    function: Any,
    args: Sequence[Any] = (),
    *,
    device: Any = None,
    kwargs: Mapping[str, Any] | None = None,
) -> Any:
    """Run a component of the release, passing `device` only when it takes one.

    The text encoders and VAEs of `diffsynth` differ in whether they accept a `device`
    argument, so it is passed when the signature carries it and dropped when the call
    rejects it.

    Args:
        function: Callable to run.
        args: Positional arguments.
        device: Device the component runs on, `None` to leave the placement to the caller.
        kwargs: Extra keyword arguments.

    Returns:
        Whatever the callable returned.
    """
    options = dict(kwargs or {})
    if device is not None:
        accepted = _signature_names(function)
        if accepted is None or "device" in accepted or "**kwargs" in accepted:
            options["device"] = torch.device(device)
    with torch.no_grad():
        try:
            return function(*args, **options)
        except TypeError:
            if not options:
                raise
            return function(*args)


def _first_tensor(output: Any) -> torch.Tensor:
    """Read a tensor out of the many shapes a pipeline output arrives in.

    Raises:
        TypeError: If the output carries no tensor at all.
    """
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


def _text_tensor(output: Any) -> torch.Tensor:
    """Read text embeddings as `(B, L, C)`.

    Raises:
        RuntimeError: If the encoder returned something that is not a batch of token
            embeddings.
    """
    tensor = _first_tensor(output)
    while tensor.dim() > 3:
        tensor = tensor[0]
    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.dim() != 3:
        raise RuntimeError(f"Expected (B, L, C) text embeddings, got {tuple(tensor.shape)}")
    return tensor


def _to_unit_range(tensor: torch.Tensor) -> torch.Tensor:
    """Map frames to `[-1, 1]`, the range the video VAE was trained on."""
    if float(tensor.min()) >= 0.0 and float(tensor.max()) > 1.5:
        return tensor / 127.5 - 1.0
    if float(tensor.min()) >= 0.0:
        return tensor * 2.0 - 1.0
    return tensor


def _as_video_batch(video: Any) -> tuple[list[torch.Tensor], bool]:
    """Split frames into the `(C, T, H, W)` clips the video VAE of the release takes.

    Args:
        video: `(T, H, W, 3)` frames, a `(C, T, H, W)` clip, a `(B, T, H, W, 3)` batch, a
            `(B, C, T, H, W)` batch or anything a float array can be read from. Values in
            `[0, 1]` or `[0, 255]` are rescaled to `[-1, 1]`.

    Returns:
        The clips and whether the input carried a leading batch dimension.

    Raises:
        ValueError: If the frames cannot be read as a clip or a batch of clips.
    """
    if torch.is_tensor(video):
        tensor = video.detach().to(dtype=torch.float32)
    else:
        array = np.asarray(video, dtype=np.float32)
        tensor = torch.from_numpy(np.ascontiguousarray(array))
    batched = tensor.dim() == 5
    if tensor.dim() == 3 and tensor.shape[-1] == 3:
        tensor = tensor.permute(2, 0, 1).unsqueeze(1)[None]
    elif tensor.dim() == 4 and tensor.shape[-1] == 3:
        tensor = tensor.permute(3, 0, 1, 2)[None]
    elif tensor.dim() == 4:
        tensor = tensor[None]
    if tensor.dim() != 5 or tensor.shape[1] != 3:
        raise ValueError(f"Expected (T, H, W, 3) frames, a (C, T, H, W) clip or a batch, got " f"{tuple(tensor.shape)}")
    tensor = _to_unit_range(tensor)
    return [clip for clip in tensor], batched


def _as_frames(output: Any) -> np.ndarray:
    """Turn a decode result into `(T, H, W, 3)` uint8 frames.

    Raises:
        ValueError: If the output cannot be read as a clip of RGB frames.
    """
    if isinstance(output, np.ndarray):
        array = np.asarray(output, dtype=np.float32)
    elif torch.is_tensor(output):
        array = output.detach().to(device="cpu", dtype=torch.float32).numpy()
    else:
        tensor = _first_tensor(output).detach().to(device="cpu", dtype=torch.float32)
        array = tensor.numpy()
    while array.ndim > 4:
        array = array[0]
    if array.ndim == 3:
        if array.shape[-1] == 3:
            array = array[None]
        elif array.shape[0] == 3:
            array = array.transpose(1, 2, 0)[None]
    if array.ndim != 4:
        raise ValueError(f"Expected generated frames, got {array.shape}")
    if array.shape[-1] != 3 and array.shape[-3] == 3:
        array = array.transpose(0, 2, 3, 1)
    if array.shape[-1] != 3:
        raise ValueError(f"Expected generated frames, got {array.shape}")
    if float(array.min()) < 0.0:
        array = (np.clip(array, -1.0, 1.0) + 1.0) * 127.5
    elif float(array.max()) <= 1.5:
        array = np.clip(array, 0.0, 1.0) * 255.0
    return np.rint(np.clip(array, 0.0, 255.0)).astype(np.uint8)


def _as_text_list(prompts: Any, expected: int | None = None) -> list[str]:
    """Read one or more prompts as a list of strings.

    Raises:
        ValueError: If no prompt was given, or the count does not match `expected`.
    """
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


def _batch_captions(batch: Any, batch_size: int) -> list[str]:
    """Read the caption of every sample of a batch.

    Raises:
        ValueError: If the batch is not a mapping.
        KeyError: If the batch carries none of the known caption keys.
    """
    if not isinstance(batch, Mapping):
        raise ValueError(f"Expected a mapping batch, got {type(batch).__name__}")
    for key in ("instruction", "caption", "prompt", "text", "task"):
        value = batch.get(key)
        if value is not None:
            return _as_text_list(value, expected=batch_size)
    raise KeyError("The batch carries no caption; expected one of instruction, caption, prompt, " "text or task")


def _first_value(mapping: Any, keys: Sequence[str]) -> Any:
    """First value a mapping holds under one of `keys`, `None` when it holds none."""
    if not isinstance(mapping, Mapping):
        return None
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            return value
    return None


def _vae_scale(vae: Any) -> int:
    """Spatial compression of the video VAE."""
    return int(getattr(vae, "upsampling_factor", VAE_SPATIAL_STRIDE))


def _latent_channels(vae: Any) -> int:
    """Channels of the video latent of the VAE."""
    return int(getattr(vae, "z_dim", LATENT_CHANNELS))


def _vae_dtype(vae: Any) -> torch.dtype | None:
    """Floating dtype of the weights of the video VAE, when it can be told."""
    declared = getattr(vae, "torch_dtype", None)
    if isinstance(declared, torch.dtype):
        return declared
    parameters = getattr(vae, "parameters", None)
    if not callable(parameters):
        return None
    for parameter in parameters():
        if parameter.is_floating_point():
            return parameter.dtype
    return None


def _encode_clip(vae: Any, clip: torch.Tensor, *, device: Any = None) -> torch.Tensor:
    """Encode one `(C, T, H, W)` clip, keeping the result on the device of the VAE.

    The VAE of the release takes a list of clips and returns `(B, C, T, h, w)`, so a single
    clip is handed over as a one-element list; a VAE that takes a batched tensor directly,
    the shape `eveworld.data.transforms.latent` wraps, is called with `clip[None]` instead.
    The clip is cast to the dtype of the VAE first, the cast `preprocess_video` of the
    release applies to the frames it encodes.

    Raises:
        RuntimeError: If the VAE exposes no `encode`.
    """
    encoder = getattr(vae, "encode", None)
    if not callable(encoder):
        raise RuntimeError("The video VAE exposes no encode(); cannot build video latents")
    target = None if device is None else torch.device(device)
    if target is not None:
        clip = clip.to(target)
    dtype = _vae_dtype(vae)
    if dtype is not None and clip.is_floating_point() and clip.dtype != dtype:
        clip = clip.to(dtype=dtype)
    accepted = _signature_names(encoder)
    with torch.no_grad():
        if accepted is None or "device" in accepted or "**kwargs" in accepted:
            try:
                output = encoder([clip], device=target)
            except TypeError:
                output = encoder(clip[None], return_dict=False)
        else:
            output = encoder(clip[None], return_dict=False)
    latent = _first_tensor(output)
    return latent[0] if latent.dim() == 5 else latent


def _decode_clip(vae: Any, latent: torch.Tensor, *, device: Any = None) -> torch.Tensor:
    """Decode one `(C, T, h, w)` latent into `(T, 3, H, W)` frames in `[-1, 1]`.

    Raises:
        RuntimeError: If the VAE exposes no `decode`.
        ValueError: If the VAE returned something that is not a clip of RGB frames.
    """
    decoder = getattr(vae, "decode", None)
    if not callable(decoder):
        raise RuntimeError("The video VAE exposes no decode(); cannot read frames")
    target = None if device is None else torch.device(device)
    if target is not None:
        latent = latent.to(target)
    dtype = _vae_dtype(vae)
    if dtype is not None and latent.is_floating_point() and latent.dtype != dtype:
        latent = latent.to(dtype=dtype)
    accepted = _signature_names(decoder)
    with torch.no_grad():
        if accepted is None or "device" in accepted or "**kwargs" in accepted:
            try:
                output = decoder([latent], device=target)
            except TypeError:
                output = decoder(latent)
        else:
            output = decoder(latent)
    frames = _first_tensor(output)
    while frames.dim() > 4:
        frames = frames[0]
    if frames.dim() != 4:
        raise ValueError(f"Expected decoded frames, got {tuple(frames.shape)}")
    if frames.shape[1] != 3 and frames.shape[-1] == 3:
        frames = frames.movedim(-1, 1)
    if frames.shape[1] != 3:
        raise ValueError(f"Expected decoded frames, got {tuple(frames.shape)}")
    return frames


def _as_pil_image(image: Any, *, height: int, width: int) -> Any:
    """Read an image as a `PIL.Image` of the requested size, the form the VAE expects.

    Args:
        image: A `PIL.Image`, an array, a tensor or a path.
        height: Height of the returned image.
        width: Width of the returned image.

    Returns:
        The image, resized to `(width, height)`.

    Raises:
        ValueError: If the input cannot be read as an image.
    """
    from PIL import Image

    if isinstance(image, Image.Image):
        return image.resize((int(width), int(height)))
    if isinstance(image, np.ndarray):
        array = image
    elif torch.is_tensor(image):
        array = image.detach().to(device="cpu", dtype=torch.float32).numpy()
    else:
        array = np.asarray(image)
    while array.ndim > 3:
        array = array[0]
    if array.ndim == 2:
        array = np.repeat(array[:, :, None], 3, axis=2)
    if array.ndim == 3 and array.shape[0] == 3 and array.shape[-1] != 3:
        array = array.transpose(1, 2, 0)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"Expected an RGB image, got {array.shape}")
    if np.issubdtype(array.dtype, np.floating):
        if float(array.min()) < 0.0:
            array = (np.clip(array, -1.0, 1.0) + 1.0) * 127.5
        elif float(array.max()) <= 1.0:
            array = np.clip(array, 0.0, 1.0) * 255.0
    array = np.clip(array, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(array).resize((int(width), int(height)))


def _white_frame(width: int, height: int) -> np.ndarray:
    """A white `(H, W, 3)` uint8 frame, the first frame of a clip without an image."""
    return np.full((int(height), int(width), 3), 255, dtype=np.uint8)


def _frame_size(pipeline: Any, height: int, width: int, num_frames: int) -> tuple[int, int]:
    """Size a pipeline resizes its input to, `(height, width)` when it does not resize.

    The pipelines of `diffsynth` round height and width up to the multiple the VAE and the
    patch embedding need, so the generation has to ask the pipeline for the size it will
    actually use, or the noise and the decoded frames disagree with the latents.
    """
    if pipeline is None:
        return int(height), int(width)
    resize = getattr(pipeline, "check_resize_height_width", None)
    if not callable(resize):
        return int(height), int(width)
    names = _signature_names(resize)
    if names is None or "num_frames" in names:
        try:
            values = resize(int(height), int(width), int(num_frames))
        except TypeError:
            values = resize(int(height), int(width))
    else:
        values = resize(int(height), int(width))
    return int(values[0]), int(values[1])


def _prefix_latents(
    vae: Any,
    image: Any,
    *,
    height: int,
    width: int,
    device: Any = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Latent of the first frame of a clip, `(1, C, 1, h, w)`.

    Args:
        vae: Video VAE that encodes the frame.
        image: First frame, as a `PIL.Image`, an array or a tensor.
        height: Frame height of the clip.
        width: Frame width of the clip.
        device: Device the VAE runs on.
        dtype: Dtype the latent is cast to.

    Returns:
        The latent of the frame, with a clip and a batch dimension.
    """
    picture = _as_pil_image(image, height=height, width=width)
    array = np.asarray(picture, dtype=np.float32) / 127.5 - 1.0
    clip = torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1)
    latent = _encode_clip(vae, clip, device=device)
    latent = latent.unsqueeze(0) if latent.dim() == 4 else latent
    if dtype is not None:
        latent = latent.to(dtype=dtype)
    return latent


def _blank_video(video: Any) -> torch.Tensor:
    """A white video with the frame count and size of `video`.

    The flow stream of a batch that carries no rendered robot-only video is trained on a
    blank stream, which keeps the two streams of the rollout in step without inventing
    motion that was never rendered. The blank stream follows the batch layout of the
    input, so a batch of clips stays a batch and lines up with the RGB stream.
    """
    clips, batched = _as_video_batch(video)
    blanks = [torch.full_like(clip, 1.0) for clip in clips]
    if batched or len(blanks) > 1:
        return torch.stack(blanks, dim=0)
    return blanks[0]


def _read_frames(path: Any, count: int, *, start: int = 0) -> np.ndarray:
    """Read `count` `(H, W, 3)` uint8 frames of a video file, starting at frame `start`.

    OpenCV is imported here so that the module stays importable without it. Videos that
    are shorter than `start + count` frames are padded by repeating their last frame.

    Args:
        path: Path of the video file.
        count: Number of frames to return.
        start: Index of the first frame of the window.

    Returns:
        The frames as `(count, H, W, 3)` uint8, in RGB order.

    Raises:
        RuntimeError: If the file cannot be opened, or holds no frame at or after `start`.
    """
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open the flow video {path}")
    frames: list[np.ndarray] = []
    index = 0
    try:
        while len(frames) < int(count):
            ok, frame = capture.read()
            if not ok:
                break
            if index >= int(start):
                frames.append(np.ascontiguousarray(frame[:, :, ::-1]))
            index += 1
    finally:
        capture.release()
    if not frames:
        raise RuntimeError(f"The flow video {path} holds no frame at or after {start}")
    while len(frames) < int(count):
        frames.append(frames[-1])
    return np.stack(frames[: int(count)])


_FLOW_TOOLS: dict[str, Any] | None = None


def _flow_tools() -> dict[str, Any]:
    """Import the flow helpers of the release checkout once, on first use.

    `flow_prefix_utils`, `raft_flow_extractor` and `reversible_flow_codec` are modules of
    the checkout rather than of the package, so they are imported lazily and cached.

    Returns:
        The `process_camera_flow` helper, the `RAFTFlowExtractor` class and the
        `FlowCodec` class.

    Raises:
        RuntimeError: If the checkout or one of the third-party packages the helpers need
            is missing.
    """
    global _FLOW_TOOLS
    if _FLOW_TOOLS is None:
        tried = _ensure_importable()
        try:
            from flow_prefix_utils import process_camera_flow
            from raft_flow_extractor import RAFTFlowExtractor
            from reversible_flow_codec import FlowCodec
        except ImportError as error:
            listing = "\n".join(f"  - {path}" for path in tried)
            raise RuntimeError(
                f"The flow helpers of the FlowWAM release could not be imported: {error}\n"
                f"looked under:\n{listing}\n"
                "clone the release with scripts/setup/clone_flowwam.sh and install "
                "requirements-flowwam.txt"
            ) from error
        _FLOW_TOOLS = {
            "process_camera_flow": process_camera_flow,
            "RAFTFlowExtractor": RAFTFlowExtractor,
            "FlowCodec": FlowCodec,
        }
    return _FLOW_TOOLS


def _flow_condition_latents(
    flow_video: Any,
    *,
    vae: Any,
    num_frames: int,
    height: int,
    width: int,
    device: Any = None,
    dtype: torch.dtype | None = None,
    start: int = 0,
) -> torch.Tensor:
    """Latents of the flow stream of a robot-only render, `(1, C, T, h, w)`.

    The render is turned into the encoded flow images the release conditions on: RAFT flow
    between consecutive frames, masked to the robot and tiled into a T-shape, which the
    codec of the release packs into RGB images. The first frame of a window is the white
    sentinel, the same convention the release uses.

    Args:
        flow_video: Path of a rendered video, or its frames.
        vae: Video VAE the flow images are encoded with, the VAE of the backbone.
        num_frames: Frames the window of the trajectory holds.
        height: Frame height of the clip.
        width: Frame width of the clip.
        device: Device the flow extractor and the VAE run on.
        dtype: Dtype the latents are cast to.
        start: Index of the window in the trajectory, the offset the render is read from.

    Returns:
        The latents of the flow stream of the window.

    Raises:
        ValueError: If the render holds fewer frames than the window needs.
        RuntimeError: If the flow helpers of the release are not importable.
    """
    tools = _flow_tools()
    if isinstance(flow_video, (str, Path)):
        frames = _read_frames(flow_video, int(num_frames), start=int(start))
    else:
        frames = _as_frames(flow_video)
        if start:
            frames = frames[int(start) :]
    if frames.shape[0] < int(num_frames):
        raise ValueError(f"The flow video holds {frames.shape[0]} frames at or after {start}, the clip " f"needs {num_frames}")
    frames = frames[: int(num_frames)]
    extractor = tools["RAFTFlowExtractor"](device=str(torch.device(device)) if device is not None else "cpu")
    pictures, _ = tools["process_camera_flow"](
        [np.ascontiguousarray(frame) for frame in frames],
        (int(width), int(height)),
        tools["FlowCodec"](),
        flow_method="raft",
        raft_extractor=extractor,
    )
    encoded = np.stack([np.asarray(picture, dtype=np.float32) for picture in pictures[: int(num_frames)]])
    encoded = encoded * (2.0 / 255.0) - 1.0
    clip = torch.from_numpy(np.ascontiguousarray(encoded)).permute(3, 0, 1, 2)
    latent = _encode_clip(vae, clip, device=device)
    latent = latent.unsqueeze(0) if latent.dim() == 4 else latent
    if dtype is not None:
        latent = latent.to(dtype=dtype)
    return latent


def _weight_map_for(
    weight_map: Any,
    grid: tuple[int, int],
    *,
    num_frames: int | None = None,
    device: Any = None,
    dtype: torch.dtype | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Resize an IGR weight map onto the latent grid and normalise it.

    The map the released fine-tuning trains under is normalised against its own moving
    part, `w / w[:, :, 1:].mean()`, so that the level of the static background does not
    set the loss scale; the map of a dataset that ships a constant one is normalised by
    that constant, which is the same as the `ones_like` fallback of the release. The
    resize is the nearest-neighbour one of `eveworld.data.transforms.latent`, which keeps
    the discrete levels of the map and reproduces the `2x2` upsampling IGR applies.

    A rank-4 map is the `(B, T, h, w)` layout collating stacked clips produces; it is
    reshaped to `(B, 1, T, h, w)` so that it broadcasts against the `(B, 1, T, h, w)`
    first-frame mask instead of right-aligning onto the channel axis.

    Args:
        weight_map: `(h, w)`, `(T, h, w)`, `(B, T, h, w)` or `(B, 1, T, h, w)` map, numpy
            or torch, on any grid.
        grid: `(height, width)` of the latent grid the map is resized onto.
        num_frames: Latent frames the map must cover; when given the map is normalised by
            the mean of its moving frames, matching the training protocol.
        device: Device of the result.
        dtype: Dtype of the result.
        eps: Floor of the normalising mean, guarding an all-zero map.

    Returns:
        The resized map, normalised, on `device` and of `dtype`.
    """
    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps}")
    from eveworld.data.transforms.latent import resize_weight_map

    if torch.is_tensor(weight_map):
        map_tensor = weight_map
    else:
        map_tensor = torch.as_tensor(np.asarray(weight_map))
    if map_tensor.dim() < 2 or map_tensor.dim() > 5:
        raise ValueError("Expected a (h, w), (T, h, w), (B, T, h, w) or (B, 1, T, h, w) weight map, got " f"{tuple(map_tensor.shape)}")
    resized = resize_weight_map(map_tensor.float(), (int(grid[0]), int(grid[1])))
    if resized.dim() == 4:
        resized = resized.unsqueeze(1)
    if resized.dim() == 5 and resized.shape[1] != 1:
        raise ValueError(f"Expected a single weight-map channel, got {resized.shape[1]}")
    axis = resized.dim() - 3
    if num_frames is not None and axis >= 0:
        length = int(resized.shape[axis])
        if length not in (1, int(num_frames)):
            raise ValueError(f"Expected a weight map of 1 or {int(num_frames)} frames, got {length}")
        weights = resized if length == 1 else resized.narrow(axis, 1, length - 1)
        resized = resized / weights.mean().clamp(min=eps)
    return resized.to(device=device, dtype=dtype)


def _batch_sigma(sigma: Any) -> torch.Tensor:
    """Noise level of a batch as a five-dimensional `(B, 1, 1, 1, 1)` tensor."""
    level = sigma if torch.is_tensor(sigma) else torch.as_tensor(sigma, dtype=torch.float32)
    return level.reshape(-1, 1, 1, 1, 1)


def _pin_first_frame(noisy: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
    """Pin the first latent frame of `noisy` to `clean` in place and return it.

    Both streams of the release are pinned this way, which makes the velocity target of
    that frame zero and keeps the conditioning frame exact through the rollout.

    Raises:
        ValueError: If the two tensors do not share a shape.
    """
    if tuple(noisy.shape) != tuple(clean.shape):
        raise ValueError(f"Expected matching shapes, got {tuple(noisy.shape)} and {tuple(clean.shape)}")
    if noisy.shape[2] > 1:
        noisy[:, :, :1] = clean[:, :, :1]
    return noisy


def _as_rank5(latents: Any) -> torch.Tensor:
    """A video latent as `(B, C, T, h, w)`, adding the batch axis to a rank-4 latent.

    Raises:
        ValueError: If the latent is not of rank 4 or 5.
    """
    tensor = latents if torch.is_tensor(latents) else torch.as_tensor(latents)
    if tensor.dim() == 4:
        return tensor[None]
    if tensor.dim() == 5:
        return tensor
    raise ValueError(f"Expected a (C, T, h, w) or (B, C, T, h, w) latent, got {tuple(tensor.shape)}")


def _randn(
    shape: Sequence[int],
    *,
    generator: torch.Generator | None = None,
    device: Any = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Gaussian noise of `shape`, drawn on the device of `generator` when one is given.

    Drawing on the device of the generator and moving the sample afterwards keeps a seed
    reproducible across devices, which is what the released inference does with its own
    `torch.Generator(device="cpu")`.
    """
    target = torch.device("cpu") if device is None else torch.device(device)
    if generator is None:
        return torch.randn(tuple(shape), device=target, dtype=dtype)
    sample = torch.randn(tuple(shape), generator=generator, device=generator.device, dtype=dtype)
    return sample.to(target)


def _flow_schedule(num_steps: int, shift: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Shifted-linear flow-matching ladder of `num_steps` steps, plus its end point.

    The ladder starts at `1.0` and ends at `0.0`, and the end point is appended again, so
    the returned tensor holds `num_steps + 1` levels and the last integration step of a
    `num_steps`-long loop advances by exactly zero. That is the trajectory of the released
    sampler, whose ladder holds `num_steps` levels whose last entry is `0.0`.

    Returns:
        `(sigmas, timesteps)`, the noise levels and the same levels scaled by the `1000`
        the DiT expects.

    Raises:
        ValueError: If `num_steps` is not positive or `shift` is not positive.
    """
    steps = int(num_steps)
    if steps < 1:
        raise ValueError(f"num_steps must be positive, got {num_steps}")
    shifted = float(shift)
    if not math.isfinite(shifted) or shifted <= 0.0:
        raise ValueError(f"shift must be a positive finite number, got {shift}")
    base = torch.linspace(1.0, 0.0, steps, dtype=torch.float32)
    sigmas = shifted * base / (1.0 + (shifted - 1.0) * base)
    sigmas = torch.cat([sigmas, torch.zeros(1, dtype=sigmas.dtype)])
    return sigmas, sigmas * _TIMESTEP_SCALE


def _train_sigmas(shift: float = 1.0, num_train_timesteps: int = 1000) -> torch.Tensor:
    """The `1000` noise levels training samples its timesteps from.

    The released fine-tuning draws `torch.randint(0, 1000, ...)` and looks the timestep up
    in the ladder of its scheduler, so index `k` of the returned tensor is the shifted
    level `shifted(k / 999)`.

    Raises:
        ValueError: If `num_train_timesteps` is not positive or `shift` is not positive.
    """
    steps = int(num_train_timesteps)
    if steps < 1:
        raise ValueError(f"num_train_timesteps must be positive, got {num_train_timesteps}")
    shifted = float(shift)
    if not math.isfinite(shifted) or shifted <= 0.0:
        raise ValueError(f"shift must be a positive finite number, got {shift}")
    base = torch.linspace(1.0, 0.0, steps, dtype=torch.float32)
    return shifted * base / (1.0 + (shifted - 1.0) * base)


def _window_starts(
    total: int,
    window: int = _WINDOW_FRAMES,
    stride: int = _WINDOW_STRIDE,
) -> list[int]:
    """First frame of every window `total` frames are generated in.

    The released long-horizon generator walks the video in `window`-frame clips that
    overlap by `window - stride` frames and pulls the last clip back so that it ends on
    the last frame, so fewer than `window` frames need a single window.

    Raises:
        ValueError: If any of the three sizes is out of range.
    """
    frames, span, step = int(total), int(window), int(stride)
    if frames < 1:
        raise ValueError(f"total must be positive, got {total}")
    if span < 2:
        raise ValueError(f"window must cover at least two frames, got {window}")
    if step < 1:
        raise ValueError(f"stride must be positive, got {stride}")
    if frames <= span:
        return [0]
    starts = [0]
    while starts[-1] + span < frames:
        following = starts[-1] + step
        if following + span > frames:
            following = frames - span
        starts.append(following)
    return starts


def _blend_windows(
    clips: Sequence[np.ndarray],
    starts: Sequence[int],
    total: int,
    crossfade: int = _CROSSFADE,
) -> np.ndarray:
    """Stitch generation windows into one video, cross-fading their overlaps.

    Each window is a `(t, H, W, 3)` clip, and consecutive windows overlap because they
    start `stride` frames apart. The overlapping part of a later window is blended onto
    the tail of the canvas with the linear alpha of the released generator, `(j + 1) /
    (crossfade + 1)` over the first `crossfade` frames of the overlap, so that the seam
    fades in from the previous window's prediction.

    Raises:
        ValueError: If a clip is not a frame stack, if a window starts before the canvas
            reaches it, or if the windows do not cover `total` frames.
    """
    if len(clips) != len(starts):
        raise ValueError(f"Expected one start per clip, got {len(clips)} and {len(starts)}")
    fade = int(crossfade)
    if fade < 0:
        raise ValueError(f"crossfade must not be negative, got {crossfade}")
    canvas: list[np.ndarray] = []
    for index, clip in enumerate(clips):
        frames = np.asarray(clip)
        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError(f"Expected a (t, H, W, 3) clip, got {tuple(frames.shape)}")
        start = int(starts[index])
        overlap = len(canvas) - start
        if overlap < 0:
            raise ValueError(f"Window {index} starts at {start} but the canvas holds {len(canvas)}")
        blended = min(fade, overlap, int(frames.shape[0]))
        for offset in range(blended):
            alpha = (offset + 1) / (fade + 1)
            head = canvas[start + offset].astype(np.float32)
            tail = frames[offset].astype(np.float32)
            mixed = (1.0 - alpha) * head + alpha * tail
            canvas[start + offset] = np.rint(np.clip(mixed, 0.0, 255.0)).astype(np.uint8)
        canvas.extend(frames[overlap:])
    if len(canvas) < int(total):
        raise ValueError(f"The windows cover {len(canvas)} frames, fewer than {int(total)}")
    return np.stack(canvas[: int(total)])


_FLOW_COND_FUNCTIONS: dict[tuple[int, int], Any] = {}
"""Cache of the flow-conditioned model function, keyed by module and source hash."""


def _dual_stream_module() -> Any:
    """Import the dual-stream pipeline module of the release.

    Raises:
        RuntimeError: If the module cannot be imported, naming every path tried.
    """
    tried = _ensure_importable()
    try:
        return importlib.import_module("diffsynth.pipelines.wan_video_dual_stream")
    except Exception as error:
        paths = ", ".join(tried) if tried else "(no candidate path)"
        raise RuntimeError(
            "Cannot import diffsynth.pipelines.wan_video_dual_stream; the FlowWAM checkout "
            f"was looked for under {paths}. Run scripts/setup/clone_flowwam.sh and point "
            "FLOWWAM_ROOT at its checkout."
        ) from error


def _dual_stream_model_fn() -> Any:
    """The `model_fn_wan_video_dual_stream` of the release.

    Raises:
        RuntimeError: If the module does not carry it, or it is not callable.
    """
    module = _dual_stream_module()
    function = getattr(module, "model_fn_wan_video_dual_stream", None)
    if not callable(function):
        raise RuntimeError(
            f"{module.__name__} carries no model_fn_wan_video_dual_stream; the FlowWAM " "checkout is not the revision this integration targets"
        )
    return function


def _flow_cond_model_fn() -> Any:
    """A variant of the released model function that zeroes the flow-stream timestep.

    Conditioning the flow stream on a rendered video means that stream is not a diffusion
    variable at all: the release patches the timestep tokens of the flow stream to zero so
    the DiT treats them as clean conditioning, exactly what `--flow-cond robot_only` of
    `inference/arm_generate.py` does. Rather than reimplementing the joint-attention
    forward pass, the statement that builds those tokens is replaced in the source of the
    released function and the result is executed in the module of that function, so every
    helper the function closes over, including the patched dual-stream block, resolves to
    the released one.

    The function is cached per module and source text, and the cache key holds the id of
    the module, so the two cache the same object only while the module is alive.

    Raises:
        RuntimeError: If the function cannot be read, if its source no longer carries the
            statement the release builds the flow timestep with, or if the generated
            variant is not callable.
    """
    module = _dual_stream_module()
    function = getattr(module, "model_fn_wan_video_dual_stream", None)
    if not callable(function):
        raise RuntimeError(
            f"{module.__name__} carries no model_fn_wan_video_dual_stream; the FlowWAM " "checkout is not the revision this integration targets"
        )
    try:
        source = inspect.getsource(function)
    except (OSError, TypeError) as error:
        raise RuntimeError(
            "Cannot read the source of model_fn_wan_video_dual_stream; run it from the "
            "FlowWAM checkout rather than a frozen or interactive interpreter"
        ) from error
    key = (id(module), hash(source))
    cached = _FLOW_COND_FUNCTIONS.get(key)
    if cached is not None:
        return cached
    start = source.find("flow_tpt = torch.cat([")
    if start < 0:
        raise RuntimeError(
            "model_fn_wan_video_dual_stream no longer builds its flow timestep with "
            "`flow_tpt = torch.cat([`; this integration cannot condition the flow stream "
            "on a rendered video for that revision"
        )
    marker = source.find("]).flatten()", start)
    if marker < 0:
        raise RuntimeError(
            "The flow timestep statement of model_fn_wan_video_dual_stream does not end " "with `]).flatten()`; the statement was not recognised"
        )
    stop = source.find("\n", marker)
    if stop < 0:
        raise RuntimeError(
            "The flow timestep statement of model_fn_wan_video_dual_stream is the last " "line of the function; the statement was not recognised"
        )
    indent = source[source.rfind("\n", 0, start) + 1 : start]
    replacement = f"{indent}flow_tpt = torch.zeros(flow_temporal * flow_spatial,\n" f"{indent}    dtype=latents.dtype, device=latents.device)"
    patched = source[:start] + replacement + source[stop:]
    patched = patched.replace("def model_fn_wan_video_dual_stream(", "def _eveworld_flow_cond_model_fn(", 1)
    try:
        code = compile(patched, "<eveworld flow-cond codegen>", "exec")
    except SyntaxError as error:
        raise RuntimeError(
            "The flow-conditioned variant of model_fn_wan_video_dual_stream does not " "compile; the released source changed shape"
        ) from error
    exec(code, module.__dict__)
    generated = module.__dict__.get("_eveworld_flow_cond_model_fn")
    if not callable(generated):
        raise RuntimeError("The flow-conditioned variant of model_fn_wan_video_dual_stream produced no " "callable")
    _FLOW_COND_FUNCTIONS[key] = generated
    return generated


def _call_dual_stream(
    model: Any,
    latents: torch.Tensor,
    flow_latents: torch.Tensor,
    timestep: Any,
    context: Any,
    *,
    flow_cond: bool = False,
    gradient_checkpointing: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run both streams through the DiT blocks of the release and return both velocities.

    `flow_cond=True` switches to the generated variant of the model function, which zeroes
    the flow-stream timestep so the flow latent conditions the sampling instead of being
    denoised, the `robot_only` mode of the released inference.

    Raises:
        RuntimeError: If the backbone carries no flow stream, or if the model function did
            not return the two velocities.
    """
    flow_stream = getattr(model, "flow_stream", None)
    if flow_stream is None:
        raise RuntimeError(
            "The backbone carries no flow stream; load a FlowWAM checkpoint with " "load_backbone() before running the dual-stream denoiser"
        )
    function = _flow_cond_model_fn() if flow_cond else _dual_stream_model_fn()
    output = function(
        dit=model.denoiser,
        flow_stream=flow_stream,
        latents=latents,
        flow_latents=flow_latents,
        timestep=torch.as_tensor(timestep).reshape(-1).to(torch.float32),
        context=torch.as_tensor(context).to(device=latents.device, dtype=latents.dtype),
        fuse_vae_embedding_in_latents=True,
        use_gradient_checkpointing=bool(gradient_checkpointing),
        use_gradient_checkpointing_offload=False,
    )
    if not isinstance(output, (tuple, list)) or len(output) != 2:
        got = type(output).__name__
        raise RuntimeError(f"Expected two velocities from the dual-stream denoiser, got {got}")
    return output[0], output[1]


def _normalise_flow_cond(value: Any) -> str:
    """`flow_cond` as one of `"none"` or `"robot_only"`.

    Raises:
        ValueError: If the value is neither.
    """
    text = str(value).strip().lower()
    if text not in {"none", "robot_only"}:
        raise ValueError(f"Unknown flow_cond {value!r}; expected 'none' or 'robot_only'")
    return text


def _normalise_full_traj(value: Any) -> str:
    """`full_traj` as one of `"direct"` or `"on"`, mapping false values to `"direct"`.

    Raises:
        ValueError: If the value names neither rollout mode.
    """
    if isinstance(value, bool):
        return "on" if value else "direct"
    text = str(value).strip().lower()
    if text in {"direct", "on"}:
        return text
    if text in {"off", "false", "none"}:
        return "direct"
    raise ValueError(f"Unknown full_traj {value!r}; expected 'direct' or 'on'")


def _normalise_switch(value: Any, name: str) -> bool:
    """A boolean flag written as a bool, a 0/1 number or the usual words.

    Raises:
        ValueError: If the value is none of those.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if value not in (0, 1):
            raise ValueError(f"{name} must be a boolean, got {value!r}")
        return bool(value)
    text = str(value).strip().lower()
    if text in {"on", "true", "yes", "1"}:
        return True
    if text in {"off", "false", "no", "0"}:
        return False
    raise ValueError(f"{name} must be a boolean, got {value!r}")


def _as_config(config: Any = None, **overrides: Any) -> FlowWAMConfig:
    """A :class:`FlowWAMConfig` out of `None`, a config or a mapping, with overrides.

    A mapping is read as a parsed run config when it carries a run-level block, and as the
    keyword fields of the dataclass otherwise, so both a `DictConfig` of
    `configs/paper/flowwam/...` and a plain field mapping can be handed in.

    Raises:
        TypeError: If the value is neither a config nor a mapping.
    """
    if config is None:
        result = FlowWAMConfig()
    elif isinstance(config, FlowWAMConfig):
        result = config
    elif isinstance(config, Mapping):
        blocks = {"model", "train", "inference", "method", "data"}
        if blocks & {str(key) for key in config}:
            result = FlowWAMConfig.from_config(config)
        else:
            result = FlowWAMConfig(**dict(config))
    else:
        raise TypeError(f"Expected a FlowWAMConfig or a mapping, got {type(config).__name__}")
    if overrides:
        result = replace(result, **overrides)
    return result


def _as_resolution(value: Any) -> tuple[int, int]:
    """`(width, height)` out of a pair of integers that are multiples of 32.

    The grid of the TIA adapter and the IGR weight map is a `2x2` patch of `16` px cells,
    so both sides of the frame have to be multiples of `32` for the geometry to close.

    Raises:
        ValueError: If the value is not a pair of positive multiples of 32.
    """
    try:
        items = tuple(int(item) for item in value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"resolution must be a pair of integers, got {value!r}") from error
    if len(items) != 2:
        raise ValueError(f"resolution must be a pair of integers, got {value!r}")
    cell = CELL_SIZE * PATCH_SIZE
    for side in items:
        if side <= 0:
            raise ValueError(f"resolution must be positive, got {value!r}")
        if side % cell != 0:
            raise ValueError(f"resolution must be a multiple of {cell}, got {value!r}")
    return (items[0], items[1])


def _infer_variant(cfg: Any) -> str:
    """Variant of a parsed run config, from `model.variant` or the names of the run.

    The robot data set is the only variant the presets describe, but a config that names
    another one keeps its value here and is rejected by the preset lookup of the dataclass
    with the list of known names.
    """
    if not isinstance(cfg, Mapping):
        return "robotwin"
    model = cfg.get("model")
    if isinstance(model, Mapping):
        declared = model.get("variant")
        if declared:
            return str(declared).strip().lower()
    data = cfg.get("data")
    parts = [str(cfg.get("output_dir", ""))]
    if isinstance(model, Mapping):
        parts.append(str(model.get("name", "")))
    if isinstance(data, Mapping):
        parts.append(str(data.get("split", "")))
        parts.append(str(data.get("metadata", "")))
    haystack = " ".join(parts).lower()
    for name in _VARIANT_PRESETS:
        if name in haystack:
            return name
    return "robotwin"


def _hooks() -> Any:
    """The hooks module of this package, imported on first use.

    The module imports this one for the geometry of the backbone, so the import is deferred
    to the call rather than run while this module is still loading.
    """
    from eveworld.integrations.flowwam import hooks

    return hooks
