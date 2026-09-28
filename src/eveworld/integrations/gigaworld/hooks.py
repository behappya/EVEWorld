"""Training-time hooks of the GigaWorld-0 integration: the IGR disturbance and the TIA term.

Three pieces sit between the clip datasets of
:mod:`eveworld.integrations.gigaworld.dataset` and
:class:`~eveworld.integrations.gigaworld.trainer.GigaWorldTrainer`:

* :func:`build_igr_transform` builds the picklable :class:`IGRTransform` that turns one clean
  item into the disturbed clip of `alg:igr_construction` plus the per-latent-frame weight map
  the restoration term consumes. It runs in the dataloader workers, so it holds no model.
* :class:`IGRCollator` wraps the transform and
  :func:`~eveworld.integrations.gigaworld.dataset.collate_fn` into the ``collate_fn`` of the
  training run, and :func:`mix_batches` folds the disturbed batch into a clean one, which is
  how the released runs keep the effective batch of 64 while perturbing only ``method.p_dup``
  of it.
* :class:`TIAHook` computes `L_TIA` from the transport-adapted features of one block forward,
  and :func:`register_tia` locates that block in a backbone and attaches the hook to it.

`L_TIA` is computed on the *post-transport* features: the query of a pair is the tracked cell of
latent frame ``t`` and the candidates are the ``window x window`` neighbourhood of the same cell
in frame ``t + 1``, so the term pulls the transport towards the correspondence the IGR annotation
marks. The positive of a pair is the tracked cell of frame ``t + 1`` at its *last* occurrence in
that window, which is the scoring the ``scatter`` based matcher of
:mod:`eveworld.methods.tia.matcher` implements; pairs whose query or target cell was not detected
are ignored. No ``(B, N, N)`` similarity matrix is ever materialised, only the ``k * k``
candidate block of one latent frame at a time.

The latent timestamp of a pixel frame is ``frame // 4``, the temporal stride of the video VAE,
and the tracked cell of a latent frame is the ``(row, column)`` entry of the ``tia_cells``
annotation of the datasets, at 16 px per cell.
"""

from __future__ import annotations

import inspect
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from eveworld.data.parsers.instruction_parser import ParsedInstruction, parse_instruction
from eveworld.data.transforms.latent import latent_grid_size, resize_weight_map
from eveworld.data.transforms.video import frames_to_tensor, tensor_to_frames
from eveworld.integrations.gigaworld.dataset import DEFAULT_SEED, _annotation, collate_fn
from eveworld.integrations.gigaworld.model import CELL_SIZE, TEMPORAL_STRIDE, latent_frames
from eveworld.methods.igr.corruption import P_DUPLICATE, IGRSample, build_sample
from eveworld.methods.igr.paste_region import STRIDE
from eveworld.methods.igr.trajectory import Track
from eveworld.methods.igr.weight_map import build_weight_map, normalize_unit_mean
from eveworld.methods.tia.adapter import TIAAdapter, TIAConfig
from eveworld.methods.tia.loss import contrastive_loss
from eveworld.methods.tia.matcher import local_window_indices, normalize_features, token_grid_shape
from eveworld.utils.io import read_json, repo_root
from eveworld.utils.logging import get_logger

__all__ = ["IGRCollator", "TIAHook", "build_igr_transform", "mix_batches", "register_tia"]

logger = get_logger(__name__)

BLOCK_CONTAINERS = ("blocks", "transformer_blocks", "dit_blocks", "layers")
"""Attribute names under which the backbones hold their blocks."""

BLOCK_SCOPES = ("dit", "denoiser", "transformer", "model")
"""Attributes that may hold the block containers one level below the backbone."""

_WINDOW_CACHE: dict[tuple[int, int, int, str], torch.Tensor] = {}


class TIAHook:
    """Forward hook of one backbone block computing the TIA term of a batch.

    The hook adapts the block output through the transport adapter and scores the tracked
    tokens of every adjacent latent frame pair, so ``L_TIA`` trains the adapter and, through
    it, the block it is attached to. Targets arrive per batch through :meth:`set_targets`;
    until they do, the hook still records the adapted features but produces no loss.

    Args:
        adapter: Transport adapter attached to the same block. The loss is computed on its
            output, i.e. on the post-transport features. ``None`` records the block features
            without adapting them, which is the measurement mode of
            :mod:`eveworld.methods.tia.layer_probe`.
        layer_index: Block index the hook is attached at, recorded on the adapter.
        grid: ``(height, width)`` token grid of the block; a three-entry sequence is read as
            its spatial part. ``None`` resolves the grid from the token count of the forward,
            e.g. ``1440`` back to ``30 x 48``.
        temperature: Override of the adapter's contrastive temperature.
        window: Override of the adapter's matching window, an odd number of cells.

    Attributes:
        adapter: The adapter in use, or ``None`` in measurement mode.
        calls: Number of forwards the hook has seen.
        features: ``(B, T, N, C)`` detached features of the last forward, or ``None``.
        last_loss: Loss of the last forward that had one, or ``None``.
        loss: Mean loss over the forwards that had one, ``0.0`` before the first.
        loss_tensor: Graph-connected scalar loss of the last forward, or ``None``. The trainer
            back-propagates through it, which is what trains the block.
        targets: The target annotation in use, or ``None``.

    Raises:
        TypeError: If ``adapter`` is neither a :class:`TIAAdapter` nor ``None``.
        ValueError: If ``window`` is not a positive odd number.
    """

    def __init__(
        self,
        adapter: TIAAdapter | None = None,
        *,
        layer_index: int | None = None,
        grid: Sequence[int] | None = None,
        temperature: float | None = None,
        window: int | None = None,
    ) -> None:
        if adapter is not None and not isinstance(adapter, TIAAdapter):
            raise TypeError(f"adapter must be a TIAAdapter or None, got {type(adapter).__name__}")
        self.adapter = adapter
        self.layer_index = None if layer_index is None else int(layer_index)
        self.grid = _as_grid(grid)
        self._handle: Any = None
        self.calls = 0
        self.features: torch.Tensor | None = None
        self.last_loss: float | None = None
        self.loss_tensor: torch.Tensor | None = None
        self.targets: torch.Tensor | None = None
        self._loss_sum = 0.0
        self._loss_count = 0
        overrides: dict[str, Any] = {}
        if temperature is not None:
            overrides["temperature"] = float(temperature)
        if window is not None:
            window = int(window)
            if window < 1 or window % 2 == 0:
                raise ValueError(f"window must be a positive odd size, got {window}")
            overrides["window"] = window
        if overrides and self.adapter is not None:
            self.adapter.config = replace(self.adapter.config, **overrides)

    @property
    def hidden_size(self) -> int | None:
        """Channel count of the block, or ``None`` in measurement mode."""
        size = getattr(self.adapter, "hidden_size", None)
        return None if size is None else int(size)

    @property
    def window(self) -> int:
        """Matching window in cells, ``7`` by default as in the paper."""
        config = getattr(self.adapter, "config", None)
        if isinstance(config, TIAConfig):
            return int(config.window)
        return int(TIAConfig().window)

    @property
    def window_radius(self) -> int:
        """Radius of the matching window, ``(window - 1) // 2``."""
        return max(0, (self.window - 1) // 2)

    @property
    def temperature(self) -> float:
        """Contrastive temperature the loss divides the scores by."""
        config = getattr(self.adapter, "config", None)
        if isinstance(config, TIAConfig):
            return float(config.temperature)
        return float(TIAConfig().temperature)

    @property
    def loss(self) -> float:
        """Mean loss over the forwards that produced one, ``0.0`` before the first."""
        return self._loss_sum / self._loss_count if self._loss_count else 0.0

    def attach(self, module: nn.Module) -> TIAHook:
        """Attach the hook to the output of ``module``.

        Args:
            module: Block module whose output carries the ``(..., hidden_size)`` features.

        Returns:
            The hook itself, for chaining.

        Raises:
            TypeError: If ``module`` is not an ``nn.Module``.
            RuntimeError: If the hook, or its adapter, is already attached.
        """
        if not isinstance(module, nn.Module):
            raise TypeError(f"module must be an nn.Module, got {type(module).__name__}")
        if self._handle is not None:
            raise RuntimeError("the TIA hook is already attached")
        if self.adapter is not None and getattr(self.adapter, "hook_handle", None) is not None:
            raise RuntimeError("the TIA adapter is already attached to a block; call remove_adapter first")
        self._handle = module.register_forward_hook(self)
        if self.adapter is not None:
            self.adapter.hook_handle = self._handle
            if self.layer_index is not None:
                self.adapter.layer_index = self.layer_index
        return self

    def detach(self) -> TIAHook:
        """Remove the hook from its module and clear the adapter's handle; idempotent."""
        handle = self._handle or getattr(self.adapter, "hook_handle", None)
        if handle is not None:
            handle.remove()
        self._handle = None
        if self.adapter is not None:
            self.adapter.hook_handle = None
            self.adapter.layer_index = None
        return self

    def set_targets(self, cells: Any = None) -> None:
        """Set the annotation the loss compares the adapted features against.

        Args:
            cells: ``(B, T, 2)`` or ``(T, 2)`` target cells ``(row, column)`` on the token
                grid, ``-1`` where the target was not detected, as the ``tia_cells`` of the
                datasets hold them; or the flat token indices ``(B, T)`` or ``(T,)`` of the
                same annotations. A single clip is broadcast over the batch. ``None`` clears
                the targets and turns the loss off.

        Raises:
            ValueError: If the annotation has an unusable shape, or if a flat index is out of
                the token grid known at the time of the call.
        """
        if cells is None:
            self.targets = None
            return
        tensor = torch.as_tensor(cells)
        if tensor.is_floating_point():
            tensor = tensor.round()
        tensor = tensor.to(torch.int64)
        if tensor.dim() == 3 and tensor.shape[-1] == 2:
            self.targets = tensor
            return
        if tensor.dim() == 2 and tensor.shape[-1] == 2:
            self.targets = tensor[None]
            return
        if tensor.dim() == 2:
            if self.grid is not None:
                _check_flat(tensor, self.grid[0] * self.grid[1])
            self.targets = tensor
            return
        if tensor.dim() == 1:
            if self.grid is not None:
                _check_flat(tensor, self.grid[0] * self.grid[1])
            self.targets = tensor[None]
            return
        raise ValueError(f"expected a (B, T, 2) cell annotation or its flat form, got {tuple(tensor.shape)}")

    def reset(self) -> None:
        """Forget the features and losses seen so far; the targets stay set."""
        self.calls = 0
        self.features = None
        self.last_loss = None
        self.loss_tensor = None
        self._loss_sum = 0.0
        self._loss_count = 0

    def __call__(self, module: nn.Module, inputs: Any, output: Any) -> Any:
        """Adapt one block forward and score its adjacent frame pairs.

        Args:
            module: The block the hook is attached to.
            inputs: Positional arguments of the block call, unused.
            output: Block output, a tensor or a tuple of tensors of which the first carries
                the features.

        Returns:
            The output with its first tensor replaced by the adapted one; a tuple keeps its
            remaining entries, and an output that is not a tensor is passed through.

        Raises:
            ValueError: If the output tensor has no axis of the adapter's hidden size.
        """
        self.calls += 1
        if isinstance(output, tuple):
            if not output:
                return output
            return (self._adapt(output[0]), *output[1:])
        return self._adapt(output)

    def _measure(self, output: torch.Tensor) -> torch.Tensor:
        """Record the features of an output without adapting them.

        Measurement mode has no hidden size to match the output against, so the layout is read
        from the token grid of the hook when it is known and defaults to the channels-first
        layouts of the block outputs. Outputs of another rank pass through unrecorded.
        """
        grid = self.grid
        if output.dim() == 5:
            if grid is not None and tuple(int(size) for size in output.shape[-3:-1]) == grid:
                features = output.reshape(output.shape[0], output.shape[1], -1, output.shape[-1])
                return self._record(output, features, grid)
            features = output.permute(0, 2, 3, 4, 1)
            features = features.reshape(output.shape[0], output.shape[2], -1, output.shape[1])
            return self._record(output, features, (int(output.shape[3]), int(output.shape[4])))
        if output.dim() == 4:
            return self._record(output, output, None)
        if output.dim() == 3:
            return self._record(output, output.unsqueeze(1), None)
        return output

    def _adapt(self, output: Any) -> Any:
        """Adapt one output tensor and record the per-token features it carries."""
        if not torch.is_tensor(output):
            return output
        hidden = self.hidden_size
        if hidden is None:
            return self._measure(output)
        if output.dim() == 5 and output.shape[1] == hidden:
            adapted = self._transport(output)
            features = adapted.permute(0, 2, 3, 4, 1)
            features = features.reshape(adapted.shape[0], adapted.shape[2], -1, adapted.shape[1])
            return self._record(adapted, features, (int(adapted.shape[3]), int(adapted.shape[4])))
        if output.dim() == 5 and output.shape[-1] == hidden:
            adapted = self._transport(output.permute(0, 4, 1, 2, 3)).permute(0, 2, 3, 4, 1)
            adapted = adapted.reshape(output.shape)
            features = adapted.reshape(adapted.shape[0], adapted.shape[1], -1, adapted.shape[-1])
            return self._record(adapted, features, (int(adapted.shape[2]), int(adapted.shape[3])))
        if output.dim() == 4 and output.shape[-1] == hidden:
            adapted = self._transport(output)
            return self._record(adapted, adapted, None)
        if output.dim() == 3 and output.shape[-1] == hidden:
            adapted = self._transport(output.unsqueeze(1)).squeeze(1)
            return self._record(adapted, adapted.unsqueeze(1), None)
        raise ValueError(
            f"cannot find the hidden size {hidden} of the TIA adapter in a block output of shape "
            f"{tuple(output.shape)}; expected (B, C, T, H, W), (B, T, H, W, C), (B, T, N, C) or "
            "(B, N, C)"
        )

    def _transport(self, tensor: torch.Tensor) -> torch.Tensor:
        """Run the adapter on ``tensor``, or pass it through in measurement mode."""
        if self.adapter is None:
            return tensor
        return self.adapter(tensor)

    def _record(
        self,
        adapted: torch.Tensor,
        features: torch.Tensor,
        grid: tuple[int, int] | None,
    ) -> torch.Tensor:
        """Keep the features of the forward and compute the loss of the batch."""
        self.features = features.detach()
        self.loss_tensor = None
        loss = self._score(features, grid)
        if loss is None:
            return adapted
        self.loss_tensor = loss
        self.last_loss = float(loss.detach())
        self._loss_sum += self.last_loss
        self._loss_count += 1
        return adapted

    def _score(self, features: torch.Tensor, grid: tuple[int, int] | None) -> torch.Tensor | None:
        """Contrastive loss of the tracked tokens over every adjacent frame pair."""
        targets = self.targets
        if targets is None or features.shape[1] < 2:
            return None
        batch, frames = int(features.shape[0]), int(features.shape[1])
        tokens = int(features.shape[2])
        resolved = grid if grid is not None else self.grid
        height, width = token_grid_shape(tokens, resolved)
        flat = _as_flat_targets(targets, height, width, batch, frames, features.device)
        normalized = normalize_features(features, dim=-1)
        windows = _window_indices(height, width, self.window_radius, features.device)
        candidates = int(windows.shape[-1])
        offsets = torch.arange(batch, device=features.device) * tokens
        positions = torch.arange(candidates, device=features.device)
        scores: list[torch.Tensor] = []
        positives: list[torch.Tensor] = []
        for frame in range(frames - 1):
            query = flat[:, frame]
            following = flat[:, frame + 1]
            usable = (query >= 0) & (following >= 0)
            if not bool(usable.any()):
                continue
            anchor = query.clamp_min(0)
            window = windows.index_select(0, anchor)
            current = normalized[:, frame].reshape(batch * tokens, -1)
            ahead = normalized[:, frame + 1].reshape(batch * tokens, -1)
            queries = current.index_select(0, anchor + offsets)
            neighbourhood = ahead.index_select(0, (window + offsets[:, None]).reshape(-1))
            similarity = (queries[:, None, :] * neighbourhood.reshape(batch, candidates, -1)).sum(-1)
            matches = window == following[:, None]
            positive = torch.where(matches, positions, torch.full_like(positions, -1)).max(dim=1).values
            usable = usable & (positive >= 0)
            if not bool(usable.any()):
                continue
            scores.append(similarity[usable])
            positives.append(positive[usable])
        if not scores:
            return None
        return contrastive_loss(
            torch.cat(scores, dim=0),
            torch.cat(positives, dim=0),
            temperature=self.temperature,
        )


def register_tia(
    model: Any,
    adapter: TIAAdapter | None = None,
    layer_index: int | None = None,
    *,
    grid: Sequence[int] | None = None,
    temperature: float | None = None,
    window: int | None = None,
) -> TIAHook:
    """Attach a :class:`TIAHook` to block ``layer_index`` of a backbone.

    The block is looked up through the containers the released backbones use (``blocks``,
    ``transformer_blocks``, ``dit_blocks``, ``layers``), first on the model itself, then on its
    ``dit``, ``denoiser``, ``transformer`` and ``model`` submodules, then by dotted name. The
    adapter's handle is set, so :func:`eveworld.methods.tia.adapter.remove_adapter` detaches the
    hook as well.

    Args:
        model: Backbone pipeline, or a wrapper such as
            :class:`~eveworld.integrations.gigaworld.model.GigaWorldModel` exposing ``backbone``.
        adapter: Transport adapter to attach with the hook. ``None`` records the features of
            the block without adapting them.
        layer_index: Block index; ``None`` reads ``model.block_index`` or
            ``model.config.block_index``, which the released configurations set to 23.
        grid: ``(height, width)`` token grid of the block; ``None`` reads ``model.grid``,
            which the GigaWorld-0 wrapper exposes as ``(frames, height, width)``.
        temperature: Override of the adapter's contrastive temperature.
        window: Override of the adapter's matching window.

    Returns:
        The attached :class:`TIAHook`.

    Raises:
        TypeError: If no block container is found on the model.
        ValueError: If ``layer_index`` cannot be resolved, or no container holds that block.
    """
    index = _layer_index(model, layer_index)
    if grid is None:
        grid = getattr(model, "grid", None)
    resolved = getattr(model, "backbone", model)
    path, module = _resolve_block(resolved, index)
    hook = TIAHook(
        adapter,
        layer_index=index,
        grid=grid,
        temperature=temperature,
        window=window,
    )
    hook.attach(module)
    logger.debug("TIA hook attached to %s", path)
    return hook


@dataclass(frozen=True)
class IGRTransform:
    """Apply the IGR disturbance of `alg:igr_construction` to one dataset item.

    The transform reads the clean clip, the target cells and the prompt of an item, builds the
    pixel-space track of the tracked instance, and hands both to
    :func:`eveworld.methods.igr.corruption.build_sample`. The weight map it returns is the one
    the restoration term consumes: a ``(T_lat, H_lat, W_lat)`` cell-grid map, unit mean over the
    volume, that marks the track and the paste region of every latent frame. An item without a
    usable annotation passes through unchanged with ``fallback`` set, which is the clean sample
    of :func:`eveworld.methods.igr.corruption.as_clean_sample`.

    Args:
        p_dup: Probability of the duplication branch of the disturbance, the ``method.p_dup``
            of the training configurations.
        seed: Base seed of the generator used when the caller passes none.
        cell_size: Pixels per annotation cell, ``16`` for the 8 px VAE and a 2 px patch.
        temporal_stride: Pixel frames per latent frame, ``4`` for the 16 FPS clips.
        stride: Lattice step of the paste-region search, in pixels.
        grid: ``(height, width)`` annotation grid; a three-entry sequence is read as its
            spatial part. ``None`` reads the grid from the item.
        metadata_dir: Directory of ``<sample_id>.json`` records, read for items that carry no
            ``tia_cells``; relative paths are anchored at the repository root.

    Raises:
        ValueError: If ``p_dup`` is not a probability, or a size or stride is not positive.
    """

    p_dup: float = P_DUPLICATE
    seed: int = DEFAULT_SEED
    cell_size: int = CELL_SIZE
    temporal_stride: int = TEMPORAL_STRIDE
    stride: int = STRIDE
    grid: tuple[int, ...] | None = None
    metadata_dir: str | os.PathLike[str] | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.p_dup) <= 1.0:
            raise ValueError(f"p_dup must be a probability, got {self.p_dup}")
        for name in ("cell_size", "temporal_stride", "stride"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")
        if self.grid is not None:
            self.grid = _as_grid(self.grid)

    def __call__(self, sample: Mapping[str, Any], rng: Any = None) -> dict[str, Any]:
        """Disturb one item and return the four keys the trainer reads.

        Args:
            sample: Dataset item holding ``video`` (or ``frames``), and ``instruction`` or
                ``caption``, plus the ``tia_cells`` annotation of the clip.
            rng: Generator, seed or ``None`` for the per-item generator of the transform.

        Returns:
            dict: ``video`` (the disturbed clip), ``weight_map`` (the ``(T_lat, H_lat, W_lat)``
            map of the disturbance), ``fallback`` and ``events``, the disturbance events of
            :class:`eveworld.methods.igr.corruption.IGREvent`. The caller merges the mapping
            into the item, which keeps its remaining keys untouched.

        Raises:
            KeyError: If the item holds neither ``video`` nor ``frames``.
            ValueError: If the clip is not a ``(T, H, W, 3)`` sequence of frames.
        """
        item = dict(sample)
        frames = _frames_of(item)
        grid = self._grid_of(item, frames)
        cells = self._cells_of(item, grid)
        parsed = parse_instruction(_instruction_of(item))
        reason = "no_track"
        track = None
        if cells is not None:
            track = _track_of(
                cells,
                parsed,
                cell_size=self.cell_size,
                num_frames=int(frames.shape[0]),
                temporal_stride=self.temporal_stride,
            )
        if track is None:
            logger.debug("IGR fell back to a clean sample: %s", reason)
            return self._clean(item, frames)
        generator = _as_rng(rng, self.seed, item)
        igr = build_sample(
            frames,
            track,
            parsed,
            p_dup=self.p_dup,
            rng=generator,
            stride=self.stride,
        )
        if igr.fallback:
            logger.debug("IGR fell back to a clean sample: %s", igr.metadata.get("reason"))
            return self._clean(item, frames)
        weight = _latent_weight_map(
            track,
            igr.events,
            grid,
            cell_size=self.cell_size,
            temporal_stride=self.temporal_stride,
            num_frames=int(frames.shape[0]),
        )
        return {
            "video": frames_to_tensor(igr.frames),
            "weight_map": weight,
            "fallback": False,
            "events": list(igr.events),
        }

    def _grid_of(self, item: Mapping[str, Any], frames: np.ndarray) -> tuple[int, int]:
        """``(height, width)`` cell grid of the item, from the configuration or its map."""
        if self.grid is not None:
            return int(self.grid[-2]), int(self.grid[-1])
        return _latent_grid(item, frames)[1:]

    def _cells_of(self, item: Mapping[str, Any], grid: tuple[int, int]) -> np.ndarray | None:
        """``(T_lat, 2)`` target cells of the item, from its annotation or a metadata record."""
        declared = item.get("tia_cells")
        if declared is not None:
            return _as_cells(declared)
        record = item.get("metadata")
        if not isinstance(record, Mapping) and self.metadata_dir is not None:
            sample_id = item.get("sample_id")
            if isinstance(sample_id, str) and sample_id:
                path = _resolve(self.metadata_dir) / f"{sample_id}.json"
                if path.is_file():
                    record = read_json(path)
        if not isinstance(record, Mapping):
            return None
        return _cells_from_record(record, grid)

    def _clean(self, item: Mapping[str, Any], frames: np.ndarray) -> dict[str, Any]:
        """The fallback result: the untouched clip behind what the item already carries."""
        weight = item.get("weight_map")
        if weight is None:
            weight = np.ones(_latent_grid(item, frames), dtype=np.float32)
        return {
            "video": item["video"] if item.get("video") is not None else frames_to_tensor(frames),
            "weight_map": weight,
            "fallback": True,
            "events": [],
        }


def build_igr_transform(
    p_dup: float = P_DUPLICATE,
    *,
    seed: int = DEFAULT_SEED,
    cell_size: int = CELL_SIZE,
    temporal_stride: int = TEMPORAL_STRIDE,
    stride: int = STRIDE,
    grid: Sequence[int] | None = None,
    metadata_dir: str | os.PathLike[str] | None = None,
) -> IGRTransform:
    """Build the IGR transform of a run.

    Args:
        p_dup: Probability of the duplication branch, ``method.p_dup``.
        seed: Base seed of the per-item generator.
        cell_size: Pixels per annotation cell, ``16`` by default.
        temporal_stride: Pixel frames per latent frame, ``4`` by default.
        stride: Lattice step of the paste-region search, in pixels.
        grid: ``(height, width)`` annotation grid; ``None`` reads it from the item.
        metadata_dir: Directory of ``<sample_id>.json`` records for items without cells.

    Returns:
        The picklable :class:`IGRTransform` the dataloader workers can hold.
    """
    return IGRTransform(
        p_dup=float(p_dup),
        seed=int(seed),
        cell_size=int(cell_size),
        temporal_stride=int(temporal_stride),
        stride=int(stride),
        grid=None if grid is None else tuple(int(value) for value in grid),
        metadata_dir=metadata_dir,
    )


class IGRCollator:
    """``collate_fn`` that disturbs every item before it is batched.

    Handed to a dataset as ``collate_fn``, this runs the IGR transform of each item with a
    per-item generator and then collates the disturbed items with
    :func:`~eveworld.integrations.gigaworld.dataset.collate_fn`. A transform that returns a
    mapping is merged into the item, one that returns an
    :class:`~eveworld.methods.igr.corruption.IGRSample` has its frames, weight map and events
    written into the item.

    Args:
        transform: Transform applied to each item; ``None`` builds the default
            :func:`build_igr_transform`.
        collate: Batch collation of the transformed items; ``None`` uses ``collate_fn``.
        seed: Base seed of the per-item generator handed to the transform.

    Attributes:
        transform: The transform in use.
        collate: The collation in use.
        seed: The base seed in use.
    """

    def __init__(
        self,
        transform: Any = None,
        *,
        collate: Any = None,
        seed: int = DEFAULT_SEED,
    ) -> None:
        self.transform = build_igr_transform() if transform is None else transform
        self.collate = collate_fn if collate is None else collate
        self.seed = int(seed)
        self._takes_rng = _takes_rng(self.transform)

    def __call__(self, batch: Sequence[Mapping[str, Any]]) -> Any:
        """Transform every item of a batch and collate the result.

        Args:
            batch: Dataset items, each a mapping.

        Returns:
            The collated batch of :meth:`collate`.
        """
        items = [self._one(item, index) for index, item in enumerate(batch)]
        return self.collate(items)

    def _one(self, item: Mapping[str, Any], index: int) -> dict[str, Any]:
        """Transform one item, merging the result into a copy of it."""
        sample = dict(item)
        rng = np.random.default_rng(self.seed + index)
        result = self.transform(sample, rng) if self._takes_rng else self.transform(sample)
        if isinstance(result, IGRSample):
            return self._sample(sample, result)
        if isinstance(result, Mapping):
            merged = dict(sample)
            merged.update(result)
            return merged
        return sample

    def _sample(self, item: dict[str, Any], sample: IGRSample) -> dict[str, Any]:
        """Write an :class:`IGRSample` into an item; a 2-D map is resized and repeated."""
        frames = np.asarray(sample.frames)
        weight = np.asarray(sample.weight_map, dtype=np.float32)
        if weight.ndim == 2:
            volume = _latent_grid(item, frames)
            weight = resize_weight_map(weight, size=(volume[1], volume[2]))
            weight = np.repeat(weight[None], volume[0], axis=0)
        merged = dict(item)
        merged["video"] = frames_to_tensor(frames)
        merged["weight_map"] = weight
        merged["fallback"] = bool(sample.fallback)
        merged["events"] = list(sample.events)
        return merged


def mix_batches(
    clean_batch: Mapping[str, Any],
    igr_batch: Mapping[str, Any],
    *,
    p_dup: float = P_DUPLICATE,
    rng: Any = None,
) -> dict[str, Any]:
    """Mix a disturbed batch into a clean one, item by item.

    The released runs keep the effective batch of 64 of the training configurations while only
    a fraction of it is disturbed: every member of the batch is drawn from ``igr_batch`` with
    probability ``p_dup`` and from ``clean_batch`` otherwise. Keys of either batch are kept; a
    key the drawn batch does not hold falls back to the other one.

    Args:
        clean_batch: Collated batch of clean clips.
        igr_batch: Collated batch of disturbed clips, of the same batch size.
        p_dup: Probability that a member is taken from ``igr_batch``.
        rng: Generator, seed or ``None`` for a fresh generator.

    Returns:
        dict: The mixed batch; empty when both batches are empty.

    Raises:
        ValueError: If the two batches hold different numbers of members, or ``p_dup`` is not a
            probability.
    """
    if not 0.0 <= float(p_dup) <= 1.0:
        raise ValueError(f"p_dup must be a probability, got {p_dup}")
    if not clean_batch and not igr_batch:
        return {}
    sizes = [size for size in (_batch_size(clean_batch), _batch_size(igr_batch)) if size is not None]
    if len(set(sizes)) > 1:
        raise ValueError(f"the batches hold different numbers of members: {sizes}")
    size = sizes[0]
    generator = _as_rng(rng, 0, None)
    take = generator.random(size) < float(p_dup)
    keys: list[str] = []
    for batch in (clean_batch, igr_batch):
        for key in batch:
            if key not in keys:
                keys.append(key)
    mixed: dict[str, Any] = {}
    for key in keys:
        values: list[Any] = []
        for index in range(size):
            if bool(take[index]) and key in igr_batch:
                values.append(_member(igr_batch[key], index, size))
            elif key in clean_batch:
                values.append(_member(clean_batch[key], index, size))
            else:
                values.append(_member(igr_batch[key], index, size))
        mixed[key] = _stack(values)
    return mixed


def _frames_of(item: Mapping[str, Any]) -> np.ndarray:
    """``(T, H, W, 3)`` uint8 frames of an item, read from ``video`` or ``frames``."""
    video = item.get("video")
    if video is not None:
        return tensor_to_frames(video)
    frames = item.get("frames")
    if frames is None:
        raise KeyError("the sample holds neither 'video' nor 'frames'")
    array = np.asarray(frames)
    if array.ndim != 4 or array.shape[-1] != 3:
        raise ValueError(f"expected a (T, H, W, 3) clip, got {array.shape}")
    return array if array.dtype == np.uint8 else np.clip(array, 0, 255).astype(np.uint8)


def _instruction_of(item: Mapping[str, Any]) -> str:
    """Instruction text of an item, falling back to its caption."""
    for key in ("instruction", "caption"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _latent_grid(item: Mapping[str, Any], frames: np.ndarray) -> tuple[int, int, int]:
    """``(T_lat, H_lat, W_lat)`` grid of an item, from its weight map or from the clip."""
    declared = item.get("weight_map")
    if declared is not None:
        shape = np.shape(declared)
        if len(shape) >= 3:
            return int(shape[-3]), int(shape[-2]), int(shape[-1])
    rows, columns = latent_grid_size((int(frames.shape[1]), int(frames.shape[2])), CELL_SIZE)
    return latent_frames(int(frames.shape[0]), TEMPORAL_STRIDE), rows, columns


def _as_cells(cells: Any) -> np.ndarray:
    """Coerce an annotation to an ``(T_lat, 2)`` int64 array of cells."""
    array = np.asarray(cells)
    if array.ndim == 1 and array.size == 2:
        array = array.reshape(1, 2)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"expected a (T, 2) cell annotation, got {array.shape}")
    return np.rint(array.astype(np.float64)).astype(np.int64)


def _cells_from_record(record: Mapping[str, Any], grid: tuple[int, int]) -> np.ndarray:
    """``(T_lat, 2)`` target cells of a metadata record, ``-1`` where undetected."""
    annotation = _annotation(record)
    cells = np.full((int(grid[0]), 2), -1, dtype=np.int64)
    if not annotation.get("gate_enabled", False):
        return cells
    entries = annotation.get("per_lat_frame")
    if not isinstance(entries, Sequence) or isinstance(entries, str):
        return cells
    for index, entry in enumerate(entries[: grid[0]]):
        if not isinstance(entry, Mapping):
            continue
        if "detected" in entry and not entry.get("detected"):
            continue
        cell = entry.get("target_cell")
        if isinstance(cell, Sequence) and not isinstance(cell, str) and len(cell) == 2:
            cells[index, 0] = int(cell[0])
            cells[index, 1] = int(cell[1])
    return cells


def _track_of(
    cells: np.ndarray,
    parsed: ParsedInstruction,
    *,
    cell_size: int,
    num_frames: int,
    temporal_stride: int,
) -> Track | None:
    """Pixel-space track of the annotated target, ``None`` when no frame holds a cell."""
    if not bool((cells >= 0).all(axis=1).any()):
        return None
    boxes = np.full((int(num_frames), 4), np.nan, dtype=np.float32)
    scores = np.zeros(int(num_frames), dtype=np.float32)
    for frame in range(int(num_frames)):
        latent = min(frame // int(temporal_stride), int(cells.shape[0]) - 1)
        row, column = int(cells[latent, 0]), int(cells[latent, 1])
        if row < 0 or column < 0:
            continue
        boxes[frame] = (
            column * cell_size,
            row * cell_size,
            (column + 1) * cell_size,
            (row + 1) * cell_size,
        )
        scores[frame] = 1.0
    label = parsed.target or (parsed.objects[0] if parsed.objects else "") or "target"
    return Track(track_id=0, label=label, boxes=boxes, scores=scores)


def _latent_weight_map(
    track: Track,
    events: Sequence[Any],
    grid: tuple[int, int],
    *,
    cell_size: int,
    temporal_stride: int,
    num_frames: int,
) -> np.ndarray:
    """``(T_lat, H_lat, W_lat)`` unit-mean weight map of a disturbed clip.

    The map mirrors what :func:`eveworld.methods.igr.corruption.build_sample` builds on the
    pixel grid, one latent frame at a time: the union of the track boxes of the four pixel
    frames the latent frame covers, or the box nearest to it when the target was not seen
    inside that window, plus the paste box of every disturbance event.
    """
    rows, columns = int(grid[0]), int(grid[1])
    boxes = np.asarray(track.boxes, dtype=np.float64) / float(cell_size)
    finite = np.isfinite(boxes).all(axis=1)
    paste = [np.asarray(event.paste_box, dtype=np.float64).reshape(1, 4) / float(cell_size) for event in events]
    layers: list[np.ndarray] = []
    for latent in range(latent_frames(int(num_frames), int(temporal_stride))):
        start = latent * int(temporal_stride)
        stop = min(start + int(temporal_stride), int(boxes.shape[0]))
        regions: list[np.ndarray] = []
        seen = [frame for frame in range(start, stop) if finite[frame]]
        if seen:
            regions.append(boxes[seen])
        elif bool(finite.any()):
            centre = (start + stop - 1) / 2.0
            nearest = int(np.argmin(np.abs(np.nonzero(finite)[0] - centre)))
            regions.append(boxes[nearest][None, :])
        regions.extend(paste)
        layers.append(build_weight_map((rows, columns), regions, normalize=False))
    return normalize_unit_mean(np.stack(layers).astype(np.float32))


def _as_rng(rng: Any, seed: int, item: Mapping[str, Any] | None) -> np.random.Generator:
    """Generator of a call: the one passed in, or a per-item one seeded by ``seed``."""
    if isinstance(rng, np.random.Generator):
        return rng
    if rng is None:
        offset = 0
        if item is not None:
            index = item.get("index")
            offset = int(index) if isinstance(index, (int, np.integer)) else 0
        return np.random.default_rng(int(seed) + offset)
    return np.random.default_rng(rng)


def _as_grid(grid: Sequence[int] | None) -> tuple[int, int] | None:
    """``(height, width)`` of a token grid; a three-entry sequence keeps its spatial part."""
    if grid is None:
        return None
    values = [int(value) for value in grid]
    if len(values) < 2:
        raise ValueError(f"expected at least a (height, width) grid, got {grid!r}")
    height, width = values[-2], values[-1]
    if height <= 0 or width <= 0:
        raise ValueError(f"grid dimensions must be positive, got {grid!r}")
    return height, width


def _check_flat(tensor: torch.Tensor, limit: int) -> None:
    """Raise if a flat target index does not fit into a grid of ``limit`` tokens."""
    if tensor.numel() and int(tensor.max()) >= int(limit):
        raise ValueError(f"flat target {int(tensor.max())} is outside the {int(limit)} tokens of the grid")


def _takes_rng(transform: Any) -> bool:
    """Whether ``transform`` accepts a second positional argument, a random generator."""
    if transform is None:
        return False
    try:
        parameters = inspect.signature(transform).parameters.values()
    except (TypeError, ValueError):
        return False
    positional = [p for p in parameters if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    return len(positional) >= 2


def _resolve(value: str | os.PathLike[str]) -> Path:
    """Resolve a configured path: expand ``~`` and anchor relative paths at the repository root."""
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else repo_root() / path


def _as_flat_targets(
    targets: torch.Tensor,
    height: int,
    width: int,
    batch: int,
    frames: int,
    device: torch.device,
) -> torch.Tensor:
    """``(B, T)`` flat token indices of an annotation, ``-1`` where undetected."""
    cells = torch.as_tensor(targets, dtype=torch.int64, device=device)
    if cells.dim() == 3:
        rows, columns = cells[..., 0], cells[..., 1]
        inside = (rows >= 0) & (columns >= 0) & (rows < int(height)) & (columns < int(width))
        flat = torch.where(inside, rows * int(width) + columns, torch.full_like(rows, -1))
    elif cells.dim() == 2:
        flat = cells
        flat = torch.where(flat >= int(height) * int(width), torch.full_like(flat, -1), flat)
    else:
        raise ValueError(f"expected a (B, T, 2) or (B, T) annotation, got {tuple(cells.shape)}")
    if flat.shape[0] == 1 and batch > 1:
        flat = flat.expand(batch, flat.shape[1])
    if flat.shape[0] != batch:
        raise ValueError(f"the annotation covers {flat.shape[0]} clips, the block holds {batch}")
    if flat.shape[1] != frames:
        raise ValueError(f"the annotation covers {flat.shape[1]} frames, the block holds {frames}")
    return flat


def _window_indices(
    height: int,
    width: int,
    radius: int,
    device: torch.device,
) -> torch.Tensor:
    """Cached ``(N, k * k)`` flat window indices of a token grid, on ``device``."""
    key = (int(height), int(width), int(radius), str(device))
    cached = _WINDOW_CACHE.get(key)
    if cached is None:
        windows = local_window_indices(int(height), int(width), int(radius))
        cached = torch.from_numpy(np.ascontiguousarray(windows))
        _WINDOW_CACHE[key] = cached
    return cached.to(device)


def _layer_index(model: Any, layer_index: int | None) -> int:
    """Block index of a run, from the argument or the ``block_index`` of the model config."""
    if layer_index is not None:
        return int(layer_index)
    for source in (model, getattr(model, "config", None)):
        value = getattr(source, "block_index", None)
        if value is not None:
            return int(value)
    raise ValueError("layer_index is required: pass the block index to register_tia")


def _resolve_block(model: Any, index: int) -> tuple[str, nn.Module]:
    """Block ``index`` of a backbone, and its dotted path.

    Args:
        model: Backbone pipeline or module.
        index: Block index, e.g. the ``model.block_index`` of the released configurations.

    Returns:
        The dotted path of the block and the block itself.

    Raises:
        TypeError: If no block container is found, or the block is not an ``nn.Module``.
        ValueError: If the containers were found but none of them holds ``index``.
    """
    notes: list[str] = []
    containers: list[tuple[str, Any]] = [(name, container) for name, container in _containers(model)]
    for scope in BLOCK_SCOPES:
        node = getattr(model, scope, None)
        if node is None or node is model:
            continue
        containers.extend((f"{scope}.{name}", container) for name, container in _containers(node))
    for path, container in containers:
        block = _container_block(container, index, path, notes)
        if block is None:
            continue
        if not isinstance(block, nn.Module):
            raise TypeError(f"block {path}[{index}] is a {type(block).__name__}, not an nn.Module")
        return f"{path}[{index}]", block
    for scope in BLOCK_SCOPES:
        node = getattr(model, scope, None)
        if node is None or node is model:
            continue
        block = _container_block(node, index, scope, notes)
        if isinstance(block, nn.Module):
            return f"{scope}[{index}]", block
    if isinstance(model, nn.Module):
        for name, module in model.named_modules():
            tail = name.rsplit(".", 1)
            if len(tail) == 2 and tail[0].rsplit(".", 1)[-1] in BLOCK_CONTAINERS:
                if tail[1] in (str(index), f"block{index}"):
                    return name, module
    if not containers and not notes:
        raise TypeError(f"no block container among {BLOCK_CONTAINERS} found on {type(model).__name__}")
    detail = "; ".join(notes) if notes else f"no container held block {index}"
    raise ValueError(f"block {index} not found on {type(model).__name__}: {detail}")


def _containers(node: Any) -> list[tuple[str, Any]]:
    """The block containers of one object, as ``(attribute name, container)`` pairs."""
    found: list[tuple[str, Any]] = []
    for name in BLOCK_CONTAINERS:
        container = getattr(node, name, None)
        if container is not None:
            found.append((name, container))
    return found


def _container_block(container: Any, index: int, path: str, notes: list[str]) -> Any | None:
    """Entry ``index`` of a container, ``None`` when the container does not hold it.

    Containers with string keys, such as a plain mapping or an ``nn.ModuleDict``, are searched
    under ``block<index>``, ``<index>`` and the integer key; sequences are indexed directly.
    """
    if isinstance(container, Mapping) or hasattr(container, "keys"):
        for key in (f"block{index}", str(index), index):
            try:
                if key in container:
                    return container[key]
            except TypeError:
                continue
        listed = ", ".join(str(key) for key in list(container)[:8])
        notes.append(f"{path} holds no block {index} (keys: {listed})")
        return None
    try:
        return container[index]
    except (IndexError, KeyError, TypeError):
        try:
            length = len(container)
        except TypeError:
            length = None
        held = "unknown length" if length is None else f"{length} blocks"
        notes.append(f"{path} holds no block {index} ({held})")
        return None


def _batch_size(batch: Mapping[str, Any]) -> int | None:
    """Number of members of a collated batch, from its tensors, arrays or lists."""
    sizes: list[int] = []
    for value in batch.values():
        if isinstance(value, (torch.Tensor, np.ndarray)):
            sizes.append(int(value.shape[0]) if value.ndim else 1)
        elif isinstance(value, (list, tuple)):
            sizes.append(len(value))
    if not sizes:
        return None
    if len(set(sizes)) > 1:
        raise ValueError(f"the batch holds values of different lengths: {sorted(set(sizes))}")
    return sizes[0]


def _member(value: Any, index: int, size: int) -> Any:
    """Member ``index`` of one batch value, repeating values that hold no batch axis."""
    if isinstance(value, (torch.Tensor, np.ndarray, list, tuple)):
        if len(value) != size:
            raise ValueError(f"a batch value holds {len(value)} members, expected {size}")
        return value[index]
    return value


def _stack(values: Sequence[Any]) -> Any:
    """Stack mixed members the way :func:`collate_fn` stacks the items of a batch."""
    if not values:
        return []
    first = values[0]
    if isinstance(first, torch.Tensor):
        return torch.stack([torch.as_tensor(value) for value in values])
    if isinstance(first, np.ndarray):
        return torch.from_numpy(np.stack([np.asarray(value) for value in values]))
    if isinstance(first, (bool, np.bool_)):
        return torch.as_tensor([bool(value) for value in values], dtype=torch.bool)
    if isinstance(first, (int, float, np.integer, np.floating)):
        return torch.as_tensor(list(values))
    return list(values)
