"""Training loop of the FlowWAM integration.

A run minimises the joint objective of `eq:pipeline`, `L = L_IGR + lambda_TIA * L_TIA`, on the
RoboTwin manipulation-transfer protocol. The restoration term is the flow-matching error of
:class:`~eveworld.integrations.flowwam.model.FlowWAMModel` on a disturbed clip, and the
transport term is the contrastive loss of
:class:`~eveworld.integrations.flowwam.hooks.FlowWAMTIAHook` on the same forward: every step
draws a clean batch and a disturbed batch of the same dataset and mixes them item by item with
:func:`~eveworld.integrations.flowwam.hooks.mix_batches`, so `method.p_dup` of
`configs/paper/flowwam/robotwin/eveworld.yaml` is the share of the batch the transport term
trains on.

Three differences separate this loop from
:class:`~eveworld.integrations.gigaworld.trainer.GigaWorldTrainer`. The backbone is fine-tuned
through LoRA adapters of rank `model.lora_rank` rather than the full denoiser: the first
:meth:`FlowWAMTrainer._prepare` of a run calls
:func:`~eveworld.integrations.flowwam.hooks.inject_lora` on the DiT and freezes every parameter
the adapters do not carry, so a run without `peft` continues with the plain flow-matching step.
The adapter sits on `model.block_index`, `12` in the released configuration, and the dual-stream
pass never calls the ``forward`` of a block, so :meth:`FlowWAMTrainer._micro_step` runs the pass
inside :func:`~eveworld.integrations.flowwam.hooks.tia_injection` instead of behind a forward
hook. The tracked cells of the annotation live on the 16 px cell grid of the IGR weight map,
twice the token grid of the denoiser, so :meth:`FlowWAMTrainer._token_cells` halves them and
clips them to the token grid the contrastive term scores on.

The released recipe trains with CAME-8bit; :func:`build_optimizer` builds it when
`came_pytorch` is importable and otherwise falls back to AdamW with the same schedule
constants, so a run without the package still steps. `scripts/train/train_flowwam.py` hands the
trainer the `model` block, the dataset and the output directory, so the trainer reads the
geometry from its configuration and the joint weights from the `method` block it is given, and
the step count from `model.max_steps`.

A released clip is 29 frames at 640x480, which cannot be stepped without the checkpoint of the
backbone. :meth:`FlowWAMTrainer.dry_run` steps the same loop at the reduced geometry
:data:`SMOKE_RESOLUTION` / :data:`SMOKE_FRAMES`, on the stand-in model of
:class:`_StandInFlowModel` when the checkpoint is not on disk, and writes nothing: it is the
wiring check of a run and the shape of the loss values a run starts from.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from eveworld.integrations.flowwam.dataset import DEFAULT_SEED, collate_fn
from eveworld.integrations.flowwam.hooks import IGRCollator, _registration, build_igr_transform, inject_lora, register_tia, tia_injection
from eveworld.integrations.flowwam.model import (
    CELL_SIZE,
    LATENT_CHANNELS,
    PATCH_SIZE,
    TEMPORAL_STRIDE,
    VAE_SPATIAL_STRIDE,
    FlowWAMConfig,
    FlowWAMModel,
    _as_config,
    _as_rank5,
    _batch_captions,
    _batch_sigma,
    _first_value,
    _pin_first_frame,
    _randn,
    latent_frames,
)
from eveworld.integrations.gigaworld.hooks import _resolve_block, mix_batches
from eveworld.methods.joint import JointConfig, JointObjective
from eveworld.methods.tia.adapter import TIAAdapter, TIAConfig, _module_hidden_size, build_adapter
from eveworld.utils.config import cfg_get, to_dict
from eveworld.utils.io import ensure_dir, is_run_checkpoint, read_json, write_json
from eveworld.utils.logging import get_logger
from eveworld.utils.seed import set_seed, worker_init_fn

__all__ = [
    "ADAPTER_DIM",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_BETAS",
    "DEFAULT_GRAD_ACCUM",
    "DEFAULT_LOG_EVERY",
    "DEFAULT_LR",
    "DEFAULT_MAX_GRAD_NORM",
    "DEFAULT_MAX_STEPS",
    "DEFAULT_SAVE_EVERY",
    "DEFAULT_WEIGHT_DECAY",
    "FlowWAMTrainer",
    "SMOKE_FRAMES",
    "SMOKE_RESOLUTION",
    "build_optimizer",
]

logger = get_logger(__name__)

DEFAULT_LR = 4.32e-5
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_BETAS = (0.9, 0.95)
DEFAULT_BATCH_SIZE = 8
DEFAULT_GRAD_ACCUM = 8
DEFAULT_MAX_STEPS = 250
DEFAULT_SAVE_EVERY = 50
DEFAULT_MAX_GRAD_NORM = 1.0
DEFAULT_LOG_EVERY = 10
ADAPTER_DIM = 64
CAME_KINDS = ("came", "came8bit", "came-8bit")
SMOKE_RESOLUTION = (160, 128)
SMOKE_FRAMES = 9


def build_optimizer(
    parameters: Any,
    *,
    kind: str = "came8bit",
    lr: float = DEFAULT_LR,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    betas: Sequence[float] = DEFAULT_BETAS,
) -> torch.optim.Optimizer:
    """Build the optimiser of a run: CAME-8bit, or AdamW when it is not available.

    The released recipe trains the LoRA adapters and the transport adapter with CAME-8bit at a
    decoupled weight decay of `0.01`. `came_pytorch` is imported lazily, and its classes take
    their hyper-parameters under different names across releases, so the keyword sets the
    package knows are tried in turn; a package that is missing, exposes neither class, or
    refuses every keyword set leaves the run on :class:`torch.optim.AdamW` with the same rate,
    decay and betas, which trains but is not the published optimiser.

    Args:
        parameters: Trainable parameters; those with ``requires_grad`` false are dropped.
        kind: Optimiser name of the run configuration, one of :data:`CAME_KINDS`, case- and
            separator-insensitive; any other name goes to AdamW.
        lr: Learning rate, `4.32e-5` in the RoboTwin run.
        weight_decay: Decoupled weight decay, `0.01` in the released configuration.
        betas: Adam betas, `(0.9, 0.95)` in the released configuration.

    Returns:
        The optimiser over the trainable parameters.

    Raises:
        ValueError: If no parameter is trainable.
    """
    values = [parameter for parameter in parameters if parameter.requires_grad]
    if not values:
        raise ValueError("no trainable parameter to optimise; build the model and the adapter before " "calling build_optimizer")
    normalized = str(kind).strip().lower().replace("-", "").replace("_", "")
    if normalized in {value.replace("-", "").replace("_", "") for value in CAME_KINDS}:
        optimiser = _came_optimizer(values, lr=lr, weight_decay=weight_decay, betas=betas)
        if optimiser is not None:
            return optimiser
    return torch.optim.AdamW(
        values,
        lr=float(lr),
        weight_decay=float(weight_decay),
        betas=(float(betas[0]), float(betas[1])),
    )


def _came_optimizer(
    parameters: Sequence[nn.Parameter],
    *,
    lr: float,
    weight_decay: float,
    betas: Sequence[float],
) -> torch.optim.Optimizer | None:
    """The CAME-8bit optimiser, or `None` when the package cannot provide one."""
    try:
        import came_pytorch  # noqa: PLC0415 - optional dependency of the released recipe
    except ImportError as error:
        logger.warning("came_pytorch is not importable (%s); falling back to AdamW", error)
        return None
    builder = getattr(came_pytorch, "CAME8bit", None)
    if not callable(builder):
        builder = getattr(came_pytorch, "CAME", None)
    if not callable(builder):
        logger.warning("came_pytorch exposes neither CAME8bit nor CAME; falling back to AdamW")
        return None
    candidates: list[dict[str, Any]] = [
        {"lr": float(lr), "weight_decay": float(weight_decay), "betas": tuple(betas)},
        {
            "lr": float(lr),
            "weight_decay": float(weight_decay),
            "beta1": float(betas[0]),
            "beta2": float(betas[1]),
        },
        {"lr": float(lr), "weight_decay": float(weight_decay)},
        {"lr": float(lr)},
    ]
    for kwargs in candidates:
        try:
            return builder(parameters, **kwargs)
        except TypeError:
            continue
        except Exception as error:  # noqa: BLE001 - any failure of the optional optimiser
            logger.warning("CAME-8bit could not be built (%s); falling back to AdamW", error)
            return None
    logger.warning(
        "the signature of %s accepted none of the known keyword sets; falling back to AdamW",
        getattr(builder, "__name__", builder),
    )
    return None


class FlowWAMTrainer:
    """Fine-tuning loop of a FlowWAM run.

    The trainer owns the pieces a run steps: the wrapped backbone
    (:class:`~eveworld.integrations.flowwam.model.FlowWAMModel`), the LoRA adapters of its DiT,
    the transport adapter of :class:`~eveworld.methods.tia.adapter.TIAAdapter`, the hook that
    routes block ``layer_index`` through the adapter, the optimiser and the joint objective. All
    of them are built lazily by :meth:`_prepare`, so a trainer can be constructed from a parsed
    configuration without the released checkpoint on disk, and only the call that trains needs
    it.

    Args:
        config: Run configuration: a
            :class:`~eveworld.integrations.flowwam.model.FlowWAMConfig`, a mapping of its
            fields, or the parsed YAML of a run such as
            `configs/paper/flowwam/robotwin/eveworld.yaml`, whose `model` block carries the
            geometry and the LoRA rank and whose `method` block carries the joint weights.
        dataset: Training dataset, e.g.
            :class:`~eveworld.integrations.flowwam.dataset.RoboTwinDataset`; `None` is only
            valid for :meth:`dry_run`.
        output_dir: Directory checkpoints and `latest.json` are written to.
        device: Device of the run; `None` uses CUDA when it is available and the CPU otherwise.
        resume: Checkpoint file or directory to resume from, or `None` for a fresh run.
        model: Backbone wrapper to train; `None` builds one from `config`.
        adapter: Transport adapter; `None` builds one for block ``layer_index``. A given
            adapter keeps its own transport configuration, so the `window` and `temperature`
            of the method block are not written into it.
        optimizer: Optimiser; `None` builds one with :func:`build_optimizer`.
        collator: Collation of the disturbed stream; `None` builds
            :class:`~eveworld.integrations.flowwam.hooks.IGRCollator` from `transform`.
        transform: IGR transform of the disturbed stream; `None` builds the default one.
        objective: Joint objective; `None` builds
            :class:`~eveworld.methods.joint.JointObjective` from `method`.
        optimizer_kind: Optimiser name, `came8bit` in the released configuration.
        lr: Learning rate.
        weight_decay: Decoupled weight decay.
        betas: Adam betas.
        batch_size: Clean and disturbed batch size; the effective batch is this times
            `grad_accum`.
        grad_accum: Micro-batches per optimisation step.
        max_steps: Optimisation steps of the run; `None` uses `config.max_steps`.
        save_every: Steps between two checkpoints; `0` disables the intermediate saves.
        num_workers: Dataloader workers of both streams.
        seed: Seed of the run.
        max_grad_norm: Gradient-norm clip; `None` disables the clip.
        log_every: Steps between two log lines; `0` logs the first step only.
        p_dup: Override of `method.p_dup`.
        lambda_tia: Override of `method.lambda_tia`.
        warmup_steps: Override of `method.warmup_steps`.
        sigma_low: Override of `method.sigma_low`.
        sigma_high: Override of `method.sigma_high`.
        gated: Override of `method.gated`.
        layer_index: Override of `method.layer`, the block the adapter sits on.
        window: Override of `method.window`, the matching window of the adapter.
        temperature: Override of `method.temperature`, the contrastive temperature.
        gamma: Override of `method.gamma`, the regularisation weight of the adapter.

    Raises:
        ValueError: If the joint weights, the window or a trainer constant are out of range.
    """

    def __init__(
        self,
        config: Any = None,
        dataset: Any = None,
        output_dir: str | Path = "outputs/run",
        device: Any = None,
        resume: str | Path | None = None,
        *,
        model: Any = None,
        adapter: Any = None,
        optimizer: Any = None,
        collator: Any = None,
        transform: Any = None,
        objective: Any = None,
        optimizer_kind: str = "came8bit",
        lr: float = DEFAULT_LR,
        weight_decay: float = DEFAULT_WEIGHT_DECAY,
        betas: Sequence[float] = DEFAULT_BETAS,
        batch_size: int = DEFAULT_BATCH_SIZE,
        grad_accum: int = DEFAULT_GRAD_ACCUM,
        max_steps: int | None = None,
        save_every: int = DEFAULT_SAVE_EVERY,
        num_workers: int = 0,
        seed: int = DEFAULT_SEED,
        max_grad_norm: float | None = DEFAULT_MAX_GRAD_NORM,
        log_every: int = DEFAULT_LOG_EVERY,
        p_dup: float | None = None,
        lambda_tia: float | None = None,
        warmup_steps: int | None = None,
        sigma_low: float | None = None,
        sigma_high: float | None = None,
        gated: bool | None = None,
        layer_index: int | None = None,
        window: int | None = None,
        temperature: float | None = None,
        gamma: float | None = None,
    ) -> None:
        self.run_config = config
        self.config = _as_config(config)
        self.dataset = dataset
        self.output_dir = Path(str(output_dir)).expanduser()
        self.device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.resume = None if resume is None else Path(str(resume)).expanduser()
        self.optimizer_kind = str(optimizer_kind)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.betas = (float(betas[0]), float(betas[1]))
        self.batch_size = int(batch_size)
        self.grad_accum = int(grad_accum)
        self.save_every = int(save_every)
        self.num_workers = int(num_workers)
        self.seed = int(seed)
        self.max_grad_norm = None if max_grad_norm is None else float(max_grad_norm)
        self.log_every = int(log_every)
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if self.grad_accum < 1:
            raise ValueError(f"grad_accum must be positive, got {grad_accum}")
        if self.save_every < 0:
            raise ValueError(f"save_every must not be negative, got {save_every}")
        self.method = JointConfig.from_mapping(
            {
                **to_dict(cfg_get(config, "method", {})),
                **{
                    name: value
                    for name, value in (
                        ("p_dup", p_dup),
                        ("lambda_tia", lambda_tia),
                        ("warmup_steps", warmup_steps),
                        ("sigma_low", sigma_low),
                        ("sigma_high", sigma_high),
                        ("gated", gated),
                    )
                    if value is not None
                },
            }
        )
        defaults = TIAConfig()
        self.layer_index = int(layer_index if layer_index is not None else cfg_get(config, "method.layer", None) or self.config.block_index)
        self.window = int(window if window is not None else cfg_get(config, "method.window", None) or defaults.window)
        self.temperature = float(temperature if temperature is not None else cfg_get(config, "method.temperature", None) or defaults.temperature)
        self.gamma = float(gamma if gamma is not None else cfg_get(config, "method.gamma", None) or defaults.gamma)
        if self.layer_index < 0:
            raise ValueError(f"layer_index must be non-negative, got {self.layer_index}")
        if self.window < 1 or self.window % 2 == 0:
            raise ValueError(f"window must be a positive odd size, got {self.window}")
        self.max_steps = int(max_steps or self.config.max_steps or DEFAULT_MAX_STEPS)
        if self.max_steps < 1:
            raise ValueError(f"max_steps must be positive, got {max_steps}")
        self.lora_rank = int(self.config.lora_rank)
        self.rng = np.random.default_rng(self.seed)
        self.generator: torch.Generator | None = None
        self.model = model
        self.adapter = adapter
        self.hook: Any = None
        self.optimizer = optimizer
        self.objective = JointObjective(self.method) if objective is None else objective
        self.collator = collator
        self.transform = transform
        self.step = 0
        self.history: list[dict[str, Any]] = []
        self._loaders_cache: tuple[DataLoader, DataLoader] | None = None
        self._stand_in = False
        self._lora = False
        self._adapter_given = adapter is not None

    def train(self) -> int:
        """Run the optimisation loop of the configuration.

        The loop steps until `max_steps`, checkpoints every `save_every` steps when it is
        positive, and returns the number of steps reached. `resume` is loaded after the model
        and the optimiser are built, so a resumed run continues from the step its checkpoint
        records.

        Returns:
            The step count of the finished run.

        Raises:
            ValueError: If the trainer was built without a dataset.
        """
        set_seed(self.seed)
        if self.dataset is None:
            raise ValueError(
                "FlowWAMTrainer.train() needs a dataset; pass one to the constructor, or use " "dry_run() to step the loop on synthetic clips"
            )
        self._prepare()
        if self.resume is not None:
            self.load_checkpoint(self.resume)
        ensure_dir(self.output_dir)
        logger.info(
            "training %s at %sx%s for %d steps, batch %d x %d accumulations",
            type(self).__name__,
            self.config.width,
            self.config.height,
            self.max_steps,
            self.batch_size,
            self.grad_accum,
        )
        stream = self._stream()
        while self.step < self.max_steps:
            self._step(stream)
            if self.save_every > 0 and self.step % self.save_every == 0:
                self.save_checkpoint()
        logger.info("finished at step %d; checkpoints in %s", self.step, self.output_dir)
        return int(self.step)

    def dry_run(
        self,
        steps: int = 1,
        allow_stand_in: bool = True,
        resolution: Sequence[int] | None = None,
        num_frames: int | None = None,
    ) -> dict[str, Any]:
        """Step the loop at the smoke geometry and write nothing.

        The run is re-pointed at :data:`SMOKE_RESOLUTION` and :data:`SMOKE_FRAMES` for the
        duration of the call — 1128 steps at the released geometry are out of reach without the
        checkpoint — and the model is rebuilt when its grid does not match the smoke one. With
        no dataset, or with `allow_stand_in` and no released checkpoint on disk, the steps run
        on the synthetic clips of :meth:`_synthetic_batch` and on the stand-in model of
        :class:`_StandInFlowModel`; the loss values of such a run check the wiring, not the
        published ones. The step counter, the history and every component the trainer held are
        restored afterwards, so the dry run leaves the trainer as it found it apart from the
        optimiser state it stepped.

        Args:
            steps: Optimisation steps to take; at least one.
            allow_stand_in: Whether the released checkpoint may be replaced by the stand-in
                model when it is not on disk.
            resolution: `(width, height)` to step at; `None` uses :data:`SMOKE_RESOLUTION`.
            num_frames: Frames of the synthetic clips; `None` uses :data:`SMOKE_FRAMES`.

        Returns:
            Mapping with the `steps` taken, `stand_in`, `device`, the `grid` and `latent_shape`
            of the smoke geometry, the mean `igr`, `tia`, `total` and `lambda_tia` of the run,
            the per-step `losses`, the number of trainable `params`, the `optimizer` class name
            and the mean `grad_norm`.

        Raises:
            ValueError: If the smoke geometry is not a multiple of the cell size, or the
                checkpoint is missing while `allow_stand_in` is false.
        """
        count = max(1, int(steps))
        smoke = replace(
            self.config,
            resolution=tuple(resolution) if resolution is not None else SMOKE_RESOLUTION,
            num_frames=int(num_frames) if num_frames is not None else SMOKE_FRAMES,
        )
        logger.warning(
            "dry run: %d step(s) at the smoke geometry %dx%d with %d frames, not the %dx%d " "with %d frames of the run",
            count,
            smoke.width,
            smoke.height,
            smoke.num_frames,
            self.config.width,
            self.config.height,
            self.config.num_frames,
        )
        saved = (
            self.config,
            self.model,
            self.adapter,
            self.hook,
            self.optimizer,
            self._loaders_cache,
            self._stand_in,
            self._lora,
        )
        history = len(self.history)
        step = self.step
        try:
            self.config = smoke
            if self.model is None or tuple(self.model.grid) != tuple(smoke.grid):
                self.model = None
                self.adapter = None
                self.hook = None
                self.optimizer = None
                self._loaders_cache = None
            self._prepare(stand_in=allow_stand_in, config=smoke)
            batch = self._synthetic_batch(smoke) if self._stand_in or self.dataset is None else next(self._stream())
            losses = [self._step(_constant_stream(batch)) for _ in range(count)]
            parameters = self._parameters()
            summary = {
                "steps": count,
                "stand_in": bool(self._stand_in),
                "device": str(self.device),
                "grid": tuple(int(value) for value in smoke.grid),
                "latent_shape": tuple(int(value) for value in smoke.latent_shape),
                "igr": _mean_of(losses, "igr"),
                "tia": _mean_of(losses, "tia"),
                "total": _mean_of(losses, "total"),
                "losses": losses,
                "lambda_tia": _mean_of(losses, "lambda_tia"),
                "params": int(sum(parameter.numel() for parameter in parameters)),
                "optimizer": type(self.optimizer).__name__,
                "grad_norm": _mean_of(losses, "grad_norm"),
            }
        finally:
            (
                self.config,
                self.model,
                self.adapter,
                self.hook,
                self.optimizer,
                self._loaders_cache,
                self._stand_in,
                self._lora,
            ) = saved
            del self.history[history:]
            self.step = step
        logger.info(
            "dry run ok: grid %s, latent %s, stand_in=%s, igr=%.4f tia=%.4f total=%.4f",
            summary["grid"],
            summary["latent_shape"],
            summary["stand_in"],
            summary["igr"],
            summary["tia"],
            summary["total"],
        )
        return summary

    def save_checkpoint(self, path: str | Path | None = None, *, step: int | None = None) -> Path:
        """Write the state of the run, and the `latest.json` pointing at it.

        The checkpoint holds the configuration, the joint weights, the denoiser state dict, the
        adapter, the optimiser, the logged history, the step counter and the state of the noise
        generator; the Python and NumPy generators are not part of it. `latest.json` sits next
        to the checkpoint and records the step and the file name, which is what
        :meth:`load_checkpoint` reads when it is given no path.

        Args:
            path: Destination: a file, or a directory, or `None` for
                `<output_dir>/checkpoint-<step:06d>.pt`.
            step: Step the checkpoint is labelled with; `None` uses the current step.

        Returns:
            The file written.
        """
        self._prepare()
        label = int(self.step if step is None else step)
        directory = ensure_dir(self.output_dir)
        target = _checkpoint_path(path, directory, label)
        ensure_dir(target.parent)
        payload = {
            "version": 1,
            "step": label,
            "seed": self.seed,
            "config": self.config.to_dict(),
            "method": asdict(self.method),
            "model": self.model.denoiser.state_dict(),
            "adapter": self.adapter.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "history": list(self.history),
            "torch_rng": None if self.generator is None else self.generator.get_state(),
        }
        torch.save(payload, target)
        write_json({"step": label, "checkpoint": target.name}, target.parent / "latest.json")
        logger.info("checkpoint of step %d written to %s", label, target)
        return target

    def load_checkpoint(self, path: str | Path | None = None) -> int:
        """Restore a run from a checkpoint and return the step it records.

        The model, the adapter and the optimiser are built first, so a checkpoint can be loaded
        into a fresh trainer that was constructed without a model. The denoiser state is loaded
        with ``strict=False`` because a checkpoint of another backbone generation may carry
        different block names; the mismatch is logged rather than raised.

        Args:
            path: Checkpoint file, a directory holding `checkpoint-*.pt` files, or `None` for
                the checkpoint `latest.json` of `output_dir` names.

        Returns:
            The step of the checkpoint.

        Raises:
            FileNotFoundError: If no checkpoint can be found at `path`.
        """
        target = self._checkpoint_file(path)
        payload = torch.load(target, map_location="cpu", weights_only=False)
        self._prepare(config=self._backbone_config(payload))
        state = payload.get("model") or {}
        if state:
            report = self.model.denoiser.load_state_dict(state, strict=False)
            missing = list(getattr(report, "missing_keys", []))
            unexpected = list(getattr(report, "unexpected_keys", []))
            if missing or unexpected:
                logger.warning(
                    "checkpoint %s does not match the model exactly: %d missing and %d " "unexpected keys",
                    target,
                    len(missing),
                    len(unexpected),
                )
        adapter = payload.get("adapter") or {}
        if adapter:
            self.adapter.load_state_dict(adapter)
        optimiser = payload.get("optimizer") or {}
        if optimiser:
            self.optimizer.load_state_dict(optimiser)
        self.history = list(payload.get("history") or [])
        self.step = int(payload.get("step", 0))
        rng = payload.get("torch_rng")
        if rng is not None and self.generator is not None:
            try:
                self.generator.set_state(torch.as_tensor(rng, dtype=torch.uint8))
            except (RuntimeError, TypeError) as error:
                logger.warning("the noise generator of %s could not be restored (%s)", target, error)
        logger.info("resumed from %s at step %d", target, self.step)
        return int(self.step)

    def _backbone_config(self, payload: Mapping[str, Any]) -> FlowWAMConfig | None:
        """The config to build the model from, when the configured checkpoint is the run.

        A run checkpoint carries the config of its run, whose `checkpoint` names the released
        backbone the weights were fine-tuned from. The inference entry points pass their
        ``--checkpoint`` through to `model.checkpoint`, so a trainer built over a
        `checkpoint-<step>.pt` would look for a released directory of that name; the recorded
        backbone is used instead. A trainer that already names a released backbone keeps its
        own config.
        """
        if not is_run_checkpoint(self.config.checkpoint):
            return None
        record = payload.get("config")
        if not isinstance(record, Mapping):
            return None
        recorded = cfg_get(record, "model.checkpoint", cfg_get(record, "checkpoint", None))
        if recorded is None or not str(recorded).strip():
            return None
        return replace(self.config, checkpoint=str(recorded).strip())

    def _checkpoint_file(self, path: str | Path | None) -> Path:
        """Resolve the checkpoint `path`, or the most recent one of the output directory."""
        if path is not None:
            candidate = Path(str(path)).expanduser()
            if candidate.is_dir():
                return _latest_checkpoint(candidate)
            if not candidate.is_file():
                raise FileNotFoundError(f"no checkpoint at {candidate}")
            return candidate
        directory = self.output_dir
        latest = directory / "latest.json"
        if latest.is_file():
            record = read_json(latest)
            name = str(record.get("checkpoint", "")) if isinstance(record, Mapping) else ""
            if name:
                recorded = latest.parent / name
                if recorded.is_file():
                    return recorded
                logger.warning("latest.json points at %s, which is not there", recorded)
        if directory.is_dir():
            return _latest_checkpoint(directory)
        raise FileNotFoundError(f"no checkpoint to resume from: {latest} is missing and {directory} does not exist")

    def _prepare(self, *, stand_in: bool = False, config: Any = None) -> None:
        """Build the model, the LoRA adapters, the transport hook and the optimiser once."""
        if config is not None:
            self.config = _as_config(config)
        if self.model is None:
            self._build_model(self.config, stand_in=stand_in)
        mover = getattr(self.model, "to", None)
        if callable(mover):
            mover(self.device)
        mode = getattr(self.model, "train", None)
        if callable(mode):
            mode()
        if not self._stand_in and not self._lora:
            denoiser = getattr(self.model, "denoiser", None)
            if isinstance(denoiser, nn.Module):
                try:
                    inject_lora(denoiser, rank=self.lora_rank)
                    self._lora = True
                    for name, parameter in denoiser.named_parameters():
                        if "lora" not in name.lower():
                            parameter.requires_grad_(False)
                except (RuntimeError, ValueError) as error:
                    logger.warning("LoRA injection failed (%s); the run continues without adapters", error)
        if self.adapter is None:
            self.adapter = self._build_adapter()
            mover = getattr(self.adapter, "to", None)
            if callable(mover):
                mover(self.device)
        if self.hook is None:
            self.hook = self._build_hook()
        if self.optimizer is None:
            self.optimizer = build_optimizer(
                self._parameters(),
                kind=self.optimizer_kind,
                lr=self.lr,
                weight_decay=self.weight_decay,
                betas=self.betas,
            )
        if self.generator is None:
            self.generator = self._make_generator()

    def _build_model(self, config: FlowWAMConfig, *, stand_in: bool = False) -> None:
        """Load the released backbone, or the stand-in model when it is not on disk."""
        self._stand_in = False
        self._lora = False
        if not stand_in:
            self.model = FlowWAMModel(config)
            return
        try:
            self.model = FlowWAMModel(config)
            return
        except (ImportError, OSError, RuntimeError, FileNotFoundError) as error:
            logger.warning(
                "the released FlowWAM backbone is not available (%s); the dry run continues " "on the stand-in model at %dx%d with %d frames",
                error,
                config.width,
                config.height,
                config.num_frames,
            )
        self._stand_in = True
        self.model = _StandInFlowModel(config)

    def _build_adapter(self) -> TIAAdapter:
        """Build the transport adapter for the hidden size of block ``layer_index``."""
        _, block = self._block()
        hidden = _module_hidden_size(block)
        if hidden is None:
            raise ValueError(
                f"cannot read the hidden size of the TIA block {self.layer_index} of "
                f"{type(block).__name__}; pass an adapter to FlowWAMTrainer(adapter=...)"
            )
        return build_adapter(
            hidden,
            TIAConfig(
                dim=ADAPTER_DIM,
                window=self.window,
                temperature=self.temperature,
                gamma=self.gamma,
                layer=self.layer_index,
            ),
        )

    def _block(self) -> tuple[str, nn.Module]:
        """The `(path, module)` of the transformer block the adapter attaches to."""
        path, module = _resolve_block(getattr(self.model, "backbone", self.model), self.layer_index)
        logger.debug("TIA block resolved to %s", path)
        return path, module

    def _build_hook(self) -> Any:
        """Register the adapter on its block and return the TIA hook of the registration."""
        grid = getattr(self.model, "tia_grid", None)
        if grid is None:
            grid = self.config.tia_grid
        register_tia(
            self.model,
            self.adapter,
            self.layer_index,
            grid=grid,
            temperature=None if self._adapter_given else self.temperature,
            window=None if self._adapter_given else self.window,
        )
        registration = _registration(self.model)
        if registration is None:
            raise RuntimeError(
                f"no TIA registration was left on {type(self.model).__name__}; the hook of " "block {self.layer_index} cannot be resolved"
            )
        return registration["hook"]

    def _make_generator(self) -> torch.Generator:
        """The deterministic generator of the noise draws of the run."""
        try:
            generator = torch.Generator(device=self.device)
        except (RuntimeError, TypeError) as error:
            logger.warning(
                "no generator on %s (%s); the noise draws run on the default one",
                self.device,
                error,
            )
            generator = torch.Generator()
        generator.manual_seed(self.seed)
        return generator

    def _parameters(self) -> list[nn.Parameter]:
        """Trainable parameters of the denoiser and of the adapter, without duplicates."""
        seen: dict[int, nn.Parameter] = {}
        for source in (self.model, self.adapter):
            if source is None:
                continue
            provider = getattr(source, "parameters", None)
            if not callable(provider):
                continue
            for parameter in provider():
                seen.setdefault(id(parameter), parameter)
        return list(seen.values())

    def _loaders(self) -> tuple[DataLoader, DataLoader]:
        """The clean and the disturbed dataloader, built once and reused across epochs.

        Both loaders iterate the same dataset with the same batch size, shuffled by their own
        generator seeded with the run seed: the two permutations are then identical, so the two
        streams stay aligned and :func:`~eveworld.integrations.flowwam.hooks.mix_batches`
        always sees batches of the same size.

        Raises:
            ValueError: If the trainer was built without a dataset.
        """
        if self._loaders_cache is not None:
            return self._loaders_cache
        if self.dataset is None:
            raise ValueError("no dataset to stream from; pass one to the constructor, or use dry_run() to " "step the loop on synthetic clips")
        collator = self.collator
        if collator is None:
            transform = self.transform
            if transform is None:
                transform = build_igr_transform(
                    self.method.p_dup,
                    seed=self.seed,
                    metadata_dir=cfg_get(self.run_config, "data.metadata", None),
                )
            collator = IGRCollator(transform, seed=self.seed)
        clean = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            worker_init_fn=worker_init_fn,
            generator=torch.Generator().manual_seed(self.seed),
        )
        disturbed = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collator,
            worker_init_fn=worker_init_fn,
            generator=torch.Generator().manual_seed(self.seed),
        )
        self._loaders_cache = (clean, disturbed)
        return self._loaders_cache

    def _stream(self) -> Iterator[dict[str, Any]]:
        """Endless stream of mixed clean and disturbed batches, one per step."""
        clean, disturbed = self._loaders()
        return (mix_batches(clean_batch, igr_batch, p_dup=self.method.p_dup, rng=self.rng) for clean_batch, igr_batch in zip(clean, disturbed))

    def _step(self, stream: Iterator[Mapping[str, Any]]) -> dict[str, Any]:
        """One optimisation step: accumulate `grad_accum` micro-batches and update."""
        self._prepare()
        started = time.perf_counter()
        self.optimizer.zero_grad(set_to_none=True)
        micros = [self._micro_step(next(stream)) for _ in range(self.grad_accum)]
        grad_norm: float | None = None
        if self.max_grad_norm is not None:
            grad_norm = float(nn.utils.clip_grad_norm_(self._parameters(), self.max_grad_norm))
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        stats = _mean_stats(micros)
        stats["grad_norm"] = grad_norm
        stats["seconds"] = time.perf_counter() - started
        stats["step"] = int(self.step)
        self.step += 1
        self.history.append(stats)
        if self.log_every > 0 and (self.step % self.log_every == 0 or self.step == 1):
            logger.info(
                "step %d/%d igr=%.4f tia=%.4f total=%.4f lambda_tia=%.4f grad_norm=%.3f %.2fs",
                self.step,
                self.max_steps,
                stats.get("igr", float("nan")),
                stats.get("tia", float("nan")),
                stats.get("total", float("nan")),
                stats.get("lambda_tia", float("nan")),
                float("nan") if grad_norm is None else grad_norm,
                stats["seconds"],
            )
        return stats

    def _micro_step(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        """One micro-batch: the joint objective of its restoration and transport terms.

        The transport term only joins the objective when the dual-stream pass of this
        micro-batch went through the adapter, which :func:`tia_injection` only arranges for the
        released backbone; the stand-in model of a dry run calls the hook itself.
        """
        self.hook.set_targets(self._token_cells(batch))
        calls = int(self.hook.calls)
        if self._stand_in:
            output = self.model.forward(batch, generator=self.generator)
        else:
            with tia_injection(self.model):
                output = self.model.forward(batch, generator=self.generator)
        igr = self.model.igr_loss(output["pred"], output["target"], output["weight_map"], output["sigma"])
        tia = self.hook.loss_tensor if int(self.hook.calls) != calls else None
        total, stats = self.objective(igr, tia, step=int(self.step), sigma=output["sigma"])
        (total / float(self.grad_accum)).backward()
        return dict(stats)

    def _token_cells(self, batch: Mapping[str, Any]) -> Any:
        """The TIA cells of a batch, on the token grid the adapter transports on.

        The annotation of the dataset lives on the `(h, w)` cell grid of the IGR weight map,
        :data:`PATCH_SIZE` times the token grid of the denoiser, while the hook scores its
        targets on the token grid; the cells are therefore halved and clipped to it. Cells of
        `-1`, the sentinel of an undetected frame, stay `-1`.
        """
        cells = batch.get("tia_cells")
        if not (torch.is_tensor(cells) and cells.dim() == 3 and cells.shape[-1] == 2):
            return cells
        rows, columns = getattr(self.model, "tia_grid", None) or self.config.tia_grid
        halved = torch.div(cells, PATCH_SIZE, rounding_mode="floor")
        limit = torch.tensor([rows - 1, columns - 1], device=cells.device, dtype=halved.dtype)
        halved = torch.minimum(halved, limit)
        return torch.where(cells >= 0, halved, cells)

    def _synthetic_batch(self, config: FlowWAMConfig) -> dict[str, Any]:
        """A batch of random clips with a valid TIA track, for :meth:`dry_run`.

        The clips are drawn at the geometry of `config` and the track stays within one cell
        around the centre of the weight-map grid, so halving it lands inside the window of the
        adapter on the token grid and the transport term is exercised instead of being skipped
        by the hook.
        """
        size = max(1, min(self.batch_size, 2))
        rng = np.random.default_rng(self.seed)
        frames = int(config.num_frames)
        video = torch.tensor(
            rng.uniform(-1.0, 1.0, (size, 3, frames, config.height, config.width)),
            dtype=torch.float32,
        )
        rows, columns = config.height // CELL_SIZE, config.width // CELL_SIZE
        latent = latent_frames(frames)
        cells = np.empty((size, latent, 2), dtype=np.int64)
        for index in range(size):
            for frame in range(latent):
                offset = int(rng.integers(-1, 2))
                cells[index, frame, 0] = min(max(rows // 2 + offset, 0), rows - 1)
                cells[index, frame, 1] = min(max(columns // 2 + offset, 0), columns - 1)
        prompts = [f"synthetic dry-run clip {index}" for index in range(size)]
        return {
            "video": video,
            "tia_cells": torch.from_numpy(cells),
            "weight_map": torch.ones((size, latent, rows, columns), dtype=torch.float32),
            "instruction": prompts,
            "caption": prompts,
        }


def _checkpoint_path(path: str | Path | None, directory: Path, step: int) -> Path:
    """Destination of a checkpoint: an explicit file, a directory, or the default name."""
    if path is None:
        return directory / f"checkpoint-{step:06d}.pt"
    candidate = Path(str(path)).expanduser()
    if candidate.is_dir() or candidate.suffix == "":
        return candidate / f"checkpoint-{step:06d}.pt"
    return candidate


def _latest_checkpoint(directory: Path) -> Path:
    """The last checkpoint of a directory, by name, which the step prefix orders."""
    found = sorted(directory.glob("checkpoint-*.pt"))
    if not found:
        raise FileNotFoundError(f"no checkpoint-*.pt file in {directory}")
    return found[-1]


def _constant_stream(batch: Mapping[str, Any]) -> Iterator[Mapping[str, Any]]:
    """Repeat one batch forever, the stream :meth:`FlowWAMTrainer.dry_run` steps on."""
    while True:
        yield batch


def _mean_of(losses: Sequence[Mapping[str, Any]], key: str) -> float:
    """Mean of a numeric key over the steps of a dry run, `nan` when it was never set."""
    values = [float(entry[key]) for entry in losses if isinstance(entry.get(key), (int, float)) and not isinstance(entry.get(key), bool)]
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def _mean_stats(entries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Average the micro-batch statistics of one step into the step statistics."""
    keys: list[str] = []
    for entry in entries:
        for key in entry:
            if key not in keys:
                keys.append(key)
    stats: dict[str, Any] = {}
    for key in keys:
        values = [entry[key] for entry in entries if key in entry]
        if key == "running":
            stats[key] = {str(name): float(value) for name, value in dict(values[-1]).items()}
            continue
        if key == "gated":
            stats[key] = all(bool(value) for value in values)
            continue
        if key == "step":
            stats[key] = int(values[-1])
            continue
        numbers = [float(value) for value in values if isinstance(value, (int, float)) and not isinstance(value, bool)]
        presented = [value for value in values if value is not None]
        if numbers and len(numbers) == len(presented):
            stats[key] = float(sum(numbers) / len(numbers))
        else:
            stats[key] = values[-1]
    return stats


class _StandInFlowBlock(nn.Module):
    """Elementwise stand-in for one dual-stream block, with a readable hidden size."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.hidden_size = int(channels)
        self.weight = nn.Parameter(torch.ones(int(channels)))
        self.bias = nn.Parameter(torch.zeros(int(channels)))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Scale and shift every token by one learned per-channel pair."""
        return hidden_states * self.weight + self.bias


class _StandInFlowDenoiser(nn.Module):
    """Patchify/blockify stand-in for the released DiT, for checkpoints that are absent.

    The forward keeps the token layout of the dual-stream pass — `(B, T, GH * GW, C)` between
    the blocks, the `(T, GH, GW)` clipping of which :class:`FlowWAMTIAHook` folds back — and
    pushes the output of the registered block through the adapter itself, because the adapter
    of a FlowWAM run is applied by :func:`~eveworld.integrations.flowwam.hooks.tia_injection`
    around the released block function and never by a forward hook.
    """

    def __init__(self, blocks: int = 24, patch: int = PATCH_SIZE, channels: int = LATENT_CHANNELS) -> None:
        super().__init__()
        self.patch_size = int(patch)
        self.channels = int(channels)
        hidden = int(channels) * int(patch) ** 2
        self.hidden_size = hidden
        self.blocks = nn.ModuleDict({f"block{index}": _StandInFlowBlock(hidden) for index in range(int(blocks))})
        self.projection = nn.Linear(hidden, hidden)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor | None = None,
        timesteps: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
        fps: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Patchify a latent, run every block and unpatchify the result."""
        batch, channels, frames, height, width = hidden_states.shape
        patch = self.patch_size
        folded = torch.nn.functional.unfold(
            hidden_states.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width),
            kernel_size=patch,
            stride=patch,
        )
        tokens = folded.transpose(1, 2).reshape(batch, frames, -1, channels * patch * patch)
        tokens = self.projection(tokens)
        registration = _registration(self)
        hook = None if registration is None else registration.get("hook")
        target = None if registration is None else registration.get("block")
        index = None if registration is None else registration.get("layer_index")
        for offset, block in enumerate(self.blocks.values()):
            tokens = block(tokens)
            if hook is not None and (block is target or offset == index):
                tokens = hook._adapt(tokens)
        unfolded = tokens.reshape(batch * frames, -1, channels, patch, patch)
        unfolded = unfolded.permute(0, 2, 3, 4, 1).reshape(batch * frames, channels, height, width)
        return unfolded.reshape(batch, frames, channels, height, width).permute(0, 2, 1, 3, 4)


class _StandInFlowVAE(nn.Module):
    """Stride-matching stand-in for the Wan video VAE of the released pipeline."""

    def __init__(
        self,
        channels: int = LATENT_CHANNELS,
        spatial: int = VAE_SPATIAL_STRIDE,
        temporal: int = TEMPORAL_STRIDE,
    ) -> None:
        super().__init__()
        self.spatial_stride = int(spatial)
        self.temporal_stride = int(temporal)
        self.project = nn.Conv3d(3, int(channels), 1)
        self.recover = nn.Conv3d(int(channels), 3, 1)

    def encode(self, clip: torch.Tensor, return_dict: bool = False) -> torch.Tensor:
        """Average the clip down to the latent grid of the backbone."""
        tensor = torch.as_tensor(clip, dtype=torch.float32)
        frames = tensor[:, :, :: self.temporal_stride]
        batch, channels, time, height, width = frames.shape
        flat = frames.reshape(batch * time, channels, height, width)
        pooled = F.avg_pool2d(flat, self.spatial_stride)
        pooled = pooled.reshape(batch, time, channels, *pooled.shape[-2:])
        return self.project(pooled.permute(0, 2, 1, 3, 4))

    def decode(self, latents: torch.Tensor, return_dict: bool = False) -> torch.Tensor:
        """Upsample a latent back to the pixel grid of the clip."""
        tensor = torch.as_tensor(latents, dtype=torch.float32)
        spread = tensor.repeat_interleave(self.temporal_stride, dim=2)
        upsampled = F.interpolate(
            spread,
            scale_factor=(1, self.spatial_stride, self.spatial_stride),
            mode="nearest",
        )
        return self.recover(upsampled)


class _StandInFlowModel(nn.Module):
    """The parts of a FlowWAM model the trainer needs, without the released weights.

    It mirrors the call surface of
    :class:`~eveworld.integrations.flowwam.model.FlowWAMModel` — the geometry properties, the
    flow-matching forward, the IGR term and the sigma draw — at a hidden size a CPU dry run
    can step.
    """

    def __init__(self, config: FlowWAMConfig) -> None:
        super().__init__()
        self.config = config
        self.denoiser = _StandInFlowDenoiser()
        self.vae = _StandInFlowVAE()

    @property
    def device(self) -> torch.device:
        """Device of the denoiser weights."""
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        """Dtype of the denoiser weights."""
        return next(self.parameters()).dtype

    @property
    def backbone(self) -> nn.Module:
        """The denoiser, under the name the block lookup understands."""
        return self.denoiser

    @property
    def grid(self) -> tuple[int, int, int]:
        """`(T, h, w)` of the latent grid, i.e. of the IGR weight map before the resize."""
        return self.config.grid

    @property
    def latent_shape(self) -> tuple[int, int, int, int]:
        """`(C, T, h, w)` of the video latent."""
        return self.config.latent_shape

    @property
    def tia_grid(self) -> tuple[int, int]:
        """`(rows, columns)` of the token grid of one latent frame."""
        return self.config.tia_grid

    def encode_video(self, video: Any, *, device: Any = None) -> torch.Tensor:
        """Encode a pixel clip into its video latent, `(B, C, T, h, w)`."""
        return _as_rank5(self.vae.encode(video))

    def encode_text(self, texts: Any, *, positive: bool = True, device: Any = None) -> torch.Tensor:
        """Deterministic pseudo-embedding of the prompts, `(B, 8, C)`."""
        prompts = [texts] if isinstance(texts, str) else list(texts)
        target = self.device if device is None else torch.device(device)
        hidden = int(self.denoiser.hidden_size)
        rows = []
        for prompt in prompts:
            digest = hashlib.sha256(str(prompt).encode("utf-8")).digest()
            values = [digest[index % len(digest)] / 255.0 - 0.5 for index in range(8 * hidden)]
            rows.append(torch.tensor(values, dtype=torch.float32).reshape(8, hidden))
        return torch.stack(rows, dim=0).to(device=target, dtype=self.dtype)

    def sample_sigma(
        self,
        batch_size: int,
        *,
        device: Any = None,
        dtype: Any = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Draw the shifted-linear noise level of a batch, as the release trains with it."""
        return FlowWAMModel.sample_sigma(self, batch_size, device=device, dtype=dtype, generator=generator)

    def igr_loss(
        self,
        pred: Any,
        target: Any,
        weight_map: Any = None,
        sigma: Any = None,
        *,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """The weighted flow-matching term of one step, the term of the release."""
        return FlowWAMModel.igr_loss(self, pred, target, weight_map, sigma, eps=eps)

    def forward(
        self,
        batch: Mapping[str, Any],
        *,
        context: Any = None,
        negative_context: Any = None,
        generator: torch.Generator | None = None,
        gradient_checkpointing: bool = False,
    ) -> dict[str, Any]:
        """One step of the flow-matching objective on the stand-in denoiser."""
        video = _first_value(batch, ("video", "frames", "rgb"))
        if video is None:
            raise KeyError("The batch carries no `video` entry to train the RGB stream on")
        target = _as_rank5(self.encode_video(video))
        source_video = _first_value(batch, ("corrected", "video_corrected", "igr_video"))
        source = target if source_video is None else _as_rank5(self.encode_video(source_video))
        batch_size = int(target.shape[0])
        sigma = self.sample_sigma(batch_size, device=target.device, dtype=torch.float32, generator=generator)
        level = _batch_sigma(sigma).to(device=target.device, dtype=target.dtype)
        noise = _randn(target.shape, generator=generator, device=target.device, dtype=target.dtype)
        noisy = _pin_first_frame((1.0 - level) * source + level * noise, source)
        if context is None:
            context = self.encode_text(_batch_captions(batch, batch_size), positive=True, device=target.device)
        timestep = sigma.reshape(-1) * 1000.0
        velocity = self.denoiser(noisy, timestep=timestep, context=context)
        return {
            "pred": noisy - level * velocity,
            "target": target,
            "weight_map": _first_value(batch, ("weight_map", "tia_weight_map", "w_map")),
            "sigma": sigma,
            "noise": noise,
            "source": source,
            "timestep": timestep,
            "pred_velocity": velocity,
            "target_velocity": noise - target,
        }
