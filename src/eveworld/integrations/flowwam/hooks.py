"""Training-time hooks of the FlowWAM integration: LoRA, the IGR disturbance and TIA.

The released FlowWAM recipe trains the Wan2.2 5B dual-stream backbone with three additions on
top of the plain flow-matching objective, and this module carries them:

* :func:`inject_lora` wraps the attention and FFN projections of the DiT with LoRA adapters of
  the configured rank, the `lora_target_modules` of the release.
* :func:`build_igr_transform`, :class:`IGRCollator` and :func:`mix_batches` are the IGR
  disturbance of the GigaWorld-0 integration, reused unchanged: its weight map lives on the
  same 16 px cells as the latent, so the weighted flow-matching term of this integration
  consumes it directly.
* :func:`register_tia`, :class:`FlowWAMTIAHook` and :func:`tia_injection` add the transport
  adapter of `sec:tia` and its InfoNCE term to one block of the denoiser.

The third piece is why this module exists rather than re-exporting the upstream hook. The
dual-stream pass of `diffsynth.pipelines.wan_video_dual_stream` never calls the ``forward`` of
the DiT blocks: its block loop calls the module-level
``_dual_stream_block_fn(block, rgb_tokens, flow_tokens, ...)``, which inlines the modulation,
attention and FFN of the block, and the gradient-checkpointing path wraps that same function.
A forward hook registered on a block would therefore never fire, and the adapter has to be
applied by replacing the module function, which is what :func:`tia_injection` does around a
pass. The checkpointing path resolves the function at call time, so the patch covers it too.

The adapter sees the RGB branch of the block output, which arrives flat as
``(B, T * GH * GW, C)``, and :class:`FlowWAMTIAHook` reshapes it through the ``(T, GH, GW)``
clip the adapter transports on. The token grid is handed over explicitly instead of being
recovered from the token count, which is ambiguous: the 2400 tokens of a 29-frame clip factor
as ``48 x 50`` rather than as the ``8 x 300`` of the released ``15 x 20`` cells per frame.

`L_TIA` is the scoring of :class:`~eveworld.integrations.gigaworld.hooks.TIAHook`, computed on
the post-transport features: the query of a pair is the tracked cell of latent frame ``t`` and
the candidates are the ``window x window`` neighbourhood of the same cell in frame ``t + 1``.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn

from eveworld.integrations.flowwam.model import CELL_SIZE, DEFAULT_LORA_RANK, TEMPORAL_STRIDE
from eveworld.integrations.gigaworld.dataset import DEFAULT_SEED
from eveworld.integrations.gigaworld.hooks import BLOCK_CONTAINERS, IGRCollator, IGRTransform, TIAHook, _as_grid, _layer_index, _resolve_block
from eveworld.integrations.gigaworld.hooks import build_igr_transform as _build_igr_transform
from eveworld.integrations.gigaworld.hooks import mix_batches
from eveworld.methods.igr.corruption import P_DUPLICATE
from eveworld.methods.igr.paste_region import STRIDE
from eveworld.methods.tia.adapter import TIAAdapter, TIAConfig, _module_hidden_size, build_adapter
from eveworld.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "DEFAULT_LORA_TARGETS",
    "FlowWAMTIAHook",
    "IGRCollator",
    "IGRTransform",
    "TIAHook",
    "build_igr_transform",
    "inject_lora",
    "mix_batches",
    "register_tia",
    "tia_injection",
    "tia_state_dict",
]


DEFAULT_LORA_TARGETS = ("q", "k", "v", "o", "ffn.0", "ffn.2")
"""Module names the released LoRA fine-tuning wraps, the `lora_target_modules` of the port."""


_REGISTRY_ATTR = "_eveworld_tia"
"""Attribute the registration of an adapter is kept under on the denoiser."""

_STATE_PREFIX = "tia_adapter."
"""Checkpoint prefix of the adapter group, as the released loaders write it."""

_STATE_ALIASES = {
    "input_proj.weight": "W_in.weight",
    "input_proj.bias": "W_in.bias",
    "output_proj.weight": "W_out.weight",
    "output_proj.bias": "W_out.bias",
}
"""Names of the ported adapter, mapped to the projection names of the method package."""

_HIDDEN_KEYS = (("W_in.weight", 1), ("W_out.weight", 0))
"""Adapter parameters that carry the hidden size, with the axis it sits on."""

_RANK_KEYS = (("W_in.weight", 0), ("W_out.weight", 1))
"""Adapter parameters that carry the low rank, with the axis it sits on."""


def inject_lora(
    backbone: nn.Module,
    rank: int = DEFAULT_LORA_RANK,
    alpha: int | None = None,
    target_modules: Sequence[str] = DEFAULT_LORA_TARGETS,
) -> nn.Module:
    """Wrap the attention and FFN projections of a DiT with LoRA adapters.

    The adapters are injected in place, as `add_lora_to_model` of
    `diffsynth.trainers.utils` does for the released fine-tuning: a
    :class:`peft.LoraConfig` of rank ``rank`` and alpha ``rank`` that targets the given module
    names. The wrapped model is returned, which is the same object for the in-place injection
    of peft.

    Args:
        backbone: The DiT to wrap, e.g. the ``denoiser`` of a
            :class:`~eveworld.integrations.flowwam.model.FlowWAMModel`.
        rank: LoRA rank, the `lora_rank` of the training configuration, ``32`` by default.
        alpha: LoRA scaling alpha; ``None`` uses ``rank``, the released default.
        target_modules: Names of the modules to wrap, relative to each DiT block.

    Returns:
        The wrapped backbone.

    Raises:
        RuntimeError: If ``peft`` is not installed.
        ValueError: If ``rank`` is not positive.
    """
    try:
        from peft import LoraConfig, inject_adapter_in_model
    except ImportError as error:
        raise RuntimeError("LoRA fine-tuning needs the peft package; install it with `pip install peft`") from error
    rank = int(rank)
    if rank < 1:
        raise ValueError(f"rank must be positive, got {rank}")
    config = LoraConfig(
        r=rank,
        lora_alpha=rank if alpha is None else int(alpha),
        target_modules=[str(name) for name in target_modules],
    )
    model = inject_adapter_in_model(config, backbone)
    logger.info("Injected LoRA of rank %d into [%s]", rank, ", ".join(config.target_modules))
    return model


class FlowWAMTIAHook(TIAHook):
    """TIA hook of the dual-stream FlowWAM backbone.

    Subclasses the upstream hook for one reason: the token stream of the dual-stream pass
    arrives flat as ``(B, T * GH * GW, C)`` and the adapter has to be run on the ``(T, GH, GW)``
    clip it transports on, with the token grid passed explicitly. The upstream hook would fold
    that layout through the token count alone and, for the 2400 tokens of a 29-frame clip,
    recover the ``48 x 50`` factorization instead of the released ``15 x 20`` cells per frame.

    The hook is never attached as a forward hook, because the block loop of
    :mod:`diffsynth.pipelines.wan_video_dual_stream` does not call the ``forward`` of a block;
    :func:`tia_injection` runs the adapter from the patched dual-stream block function instead.

    Args:
        adapter: Transport adapter the term trains; ``None`` records the block features
            without adapting them.
        layer_index: Block index the adapter belongs to.
        grid: ``(height, width)`` token grid of the block, e.g. ``(15, 20)`` for the released
            geometry; ``None`` falls back to the token factorization of the upstream hook.
        temperature: Override of the adapter's contrastive temperature.
        window: Override of the adapter's matching window.
    """

    def attach(self, module: nn.Module) -> FlowWAMTIAHook:
        """Refuse the forward-hook route, which the dual-stream pass never triggers.

        Args:
            module: Unused.

        Raises:
            RuntimeError: Always; run the pass inside :func:`tia_injection` instead.
        """
        raise RuntimeError(
            "a forward hook on a FlowWAM block never fires: the dual-stream pass calls "
            "diffsynth.pipelines.wan_video_dual_stream._dual_stream_block_fn rather than the "
            "forward of the block; run the pass inside tia_injection(...) instead"
        )

    def _adapt(self, output: Any) -> Any:
        """Adapt one block-output token stream and record its per-token features.

        A flat ``(B, T * GH * GW, C)`` or ``(B, T, GH * GW, C)`` stream whose token count
        divides by the known token grid is folded back to the ``(T, GH, GW)`` clip of the
        adapter; every other layout is handed to the upstream hook. The call is counted here
        rather than in ``TIAHook.__call__``, because the dual-stream pass and the stand-in
        model of a dry run route their block outputs through this method directly.
        """
        if not torch.is_tensor(output):
            return output
        self.calls += 1
        hidden = self.hidden_size
        if hidden is None:
            return self._measure(output)
        folded = _as_stream_volume(output, hidden, self.grid)
        if folded is None:
            return super()._adapt(output)
        volume, frames, height, width = folded
        adapted = self._transport(volume).permute(0, 2, 3, 4, 1).reshape(output.shape)
        features = adapted.reshape(adapted.shape[0], frames, height * width, hidden)
        return self._record(adapted, features, (height, width))


def register_tia(
    model: Any,
    adapter: TIAAdapter | None = None,
    layer_index: int | None = None,
    *,
    grid: Sequence[int] | None = None,
    temperature: float | None = None,
    window: int | None = None,
    state_dict: Mapping[str, Any] | None = None,
    rank: int | None = None,
    gamma: float | None = None,
) -> TIAAdapter | None:
    """Register a TIA adapter on block ``layer_index`` of a FlowWAM denoiser.

    Unlike the upstream ``register_tia`` this does not attach a forward hook: it locates the
    block, builds or adopts the adapter, loads the checkpoint entries that come with it and
    records all of it on the denoiser, from where :func:`tia_injection` routes the dual-stream
    passes through the adapter and :func:`tia_state_dict` writes it back out.

    Args:
        model: A :class:`~eveworld.integrations.flowwam.model.FlowWAMModel`, a released
            pipeline carrying a ``dit`` or ``denoiser``, or the denoiser itself.
        adapter: Adapter to register. ``None`` builds one from ``state_dict`` or from the
            geometry of the backbone.
        layer_index: Block index; ``None`` reads ``block_index`` from the model or its config,
            which the released configuration sets to ``12``.
        grid: ``(height, width)`` token grid of the annotation, e.g. ``(15, 20)``; ``None``
            reads ``tia_grid`` from the denoiser or its config.
        temperature: Override of the adapter's contrastive temperature.
        window: Override of the adapter's matching window, an odd number of cells.
        state_dict: Adapter weights, keyed either with the projection names of this package or
            with the ``input_proj`` / ``output_proj`` names of the ported checkpoint, and
            optionally carrying the ``tia_adapter.`` prefix of a checkpoint.
        rank: Low rank of a new adapter; ``None`` reads it from ``state_dict`` and falls back to
            the ``64`` of the method defaults.
        gamma: Residual scale of a new adapter; ``None`` uses the method default.

    Returns:
        The registered adapter, or ``None`` if no adapter could be built.

    Raises:
        ValueError: If no block index is given or exposed, if the hidden size of the block
            cannot be determined, or if ``window`` is not a positive odd number.
        TypeError: If no block container is found on the denoiser.
    """
    denoiser = _denoiser_of(model)
    index = _layer_index(denoiser, layer_index)
    token_grid = _as_grid(grid) if grid is not None else _token_grid(denoiser)
    path, block = _resolve_block(denoiser, index)
    state = _normalise_state(state_dict)
    if adapter is None:
        hidden = _hidden_size(state=state, block=block, denoiser=denoiser)
        if hidden is None:
            raise ValueError(
                f"cannot determine the hidden size of block {path}; pass an adapter, a " "state_dict or a model exposing `dim`/`hidden_size`"
            )
        config = TIAConfig(
            dim=_rank(rank, state),
            window=TIAConfig().window if window is None else int(window),
            temperature=TIAConfig().temperature if temperature is None else float(temperature),
            gamma=TIAConfig().gamma if gamma is None else float(gamma),
            layer=index,
        )
        adapter = build_adapter(hidden, config)
        logger.info("Built a TIA adapter of rank %d for the %d channels of %s", adapter.dim, hidden, path)
    if state:
        _load_state(adapter, state)
    hook = FlowWAMTIAHook(adapter, layer_index=index, grid=token_grid, temperature=temperature, window=window)
    registration = {
        "adapter": adapter,
        "hook": hook,
        "layer_index": index,
        "grid": token_grid,
        "block": block,
        "path": path,
        "n_blocks": _block_count(denoiser),
    }
    setattr(denoiser, _REGISTRY_ATTR, registration)
    logger.info(
        "Registered a TIA adapter on %s with a %s token grid",
        path,
        "unknown" if token_grid is None else f"{token_grid[0]}x{token_grid[1]}",
    )
    return adapter


def tia_state_dict(model: Any) -> dict[str, torch.Tensor]:
    """Adapter weights of a model, keyed as a checkpoint writes them.

    Args:
        model: A :class:`~eveworld.integrations.flowwam.model.FlowWAMModel`, a released
            pipeline, or the denoiser a TIA adapter is registered on.

    Returns:
        dict: The detached adapter weights under the ``tia_adapter.`` prefix, the group
            :meth:`FlowWAMModel.state_dict` writes and :func:`register_tia` reads back; empty
            when no adapter is registered.
    """
    registration = _registration(model)
    if registration is None:
        return {}
    adapter = registration.get("adapter")
    if not isinstance(adapter, nn.Module):
        return {}
    return {f"{_STATE_PREFIX}{key}": value.detach() for key, value in adapter.state_dict().items()}


@contextlib.contextmanager
def tia_injection(model: Any, enabled: bool = True) -> Iterator[FlowWAMTIAHook | None]:
    """Route the dual-stream passes of a pass through the registered TIA adapter.

    Replaces ``_dual_stream_block_fn`` of the dual-stream pipeline module for the duration of
    the context: every call runs the released block function first and, where the call is the
    registered block, pushes the RGB token stream through the adapter. The counter of the
    wrapper falls back to ``index % n_blocks`` when the released block objects cannot be
    recognised, which keeps the adapter hit in the repeated forwards of a sampling loop.

    Nothing is patched when ``enabled`` is false or when no adapter is registered, so an
    inference pass with ``tia_inject`` off pays no import of the pipeline module either.

    Args:
        model: The model or denoiser the adapter was registered on.
        enabled: Whether the adapter should take part in the pass.

    Yields:
        The hook of the registered adapter, or ``None`` when the adapter stays inactive; the
        hook carries the features and the ``loss_tensor`` of the last forwarded block.
    """
    registration = _registration(model)
    hook = None if registration is None else registration.get("hook")
    if not enabled:
        yield None
        return
    if not isinstance(hook, FlowWAMTIAHook):
        logger.warning(
            "tia_injection is enabled but no TIA adapter is registered on the model; call " "register_tia first. The pass runs without the adapter."
        )
        yield None
        return
    from eveworld.integrations.flowwam import model as flowwam_model

    try:
        module = flowwam_model._dual_stream_module()
    except RuntimeError as error:
        logger.warning("Cannot inject TIA into the dual-stream pass: %s", error)
        yield None
        return
    original = getattr(module, "_dual_stream_block_fn", None)
    if not callable(original):
        logger.warning(
            "%s carries no _dual_stream_block_fn; the TIA adapter stays inactive in this pass",
            module.__name__,
        )
        yield None
        return
    target = registration.get("block")
    n_blocks = registration.get("n_blocks")
    index = registration.get("layer_index")
    counters = {"calls": 0}

    def wrapped(block: Any, rgb_tokens: Any, flow_tokens: Any, *args: Any, **kwargs: Any) -> Any:
        result = original(block, rgb_tokens, flow_tokens, *args, **kwargs)
        hit = block is target
        if not hit and isinstance(n_blocks, int) and n_blocks > 0 and isinstance(index, int):
            hit = counters["calls"] % n_blocks == index
        counters["calls"] += 1
        if not hit or not isinstance(result, tuple) or len(result) != 2:
            return result
        rgb, flow = result
        if not torch.is_tensor(rgb):
            return result
        return hook._adapt(rgb), flow

    module._dual_stream_block_fn = wrapped
    try:
        yield hook
    finally:
        if getattr(module, "_dual_stream_block_fn", None) is wrapped:
            module._dual_stream_block_fn = original
        else:
            logger.debug("The dual-stream block function was replaced during the TIA context")


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
    """Build the IGR transform of a FlowWAM run.

    The disturbance is the one of the GigaWorld-0 integration, with the defaults pinned to the
    geometry of this backbone: a clip is 29 latent frames of 16 px cells and the latent frame
    of a pixel frame is ``frame // 4``. The transform runs in the dataloader workers, so it
    holds no model.

    Args:
        p_dup: Probability of the duplication branch, ``method.p_dup``.
        seed: Base seed of the per-item generator.
        cell_size: Pixels per annotation cell, ``16`` for this backbone.
        temporal_stride: Pixel frames per latent frame, ``4`` for the video VAE.
        stride: Lattice step of the paste-region search, in pixels.
        grid: ``(height, width)`` annotation grid, e.g. ``(30, 40)``; ``None`` reads it from
            the item.
        metadata_dir: Directory of ``<sample_id>.json`` records for items without cells.

    Returns:
        The picklable :class:`IGRTransform` the dataloader workers can hold.
    """
    return _build_igr_transform(
        p_dup,
        seed=seed,
        cell_size=cell_size,
        temporal_stride=temporal_stride,
        stride=stride,
        grid=grid,
        metadata_dir=metadata_dir,
    )


def _as_stream_volume(
    output: torch.Tensor,
    hidden: int,
    grid: Sequence[int] | None,
) -> tuple[torch.Tensor, int, int, int] | None:
    """``(B, C, T, H, W)`` view of a token stream on a known token grid, or ``None``.

    Args:
        output: Flat ``(B, T * H * W, C)`` or ``(B, T, H * W, C)`` block output.
        hidden: Channel count the last axis must carry.
        grid: ``(height, width)`` token grid, or ``None`` to refuse the conversion.

    Returns:
        The ``(B, C, T, H, W)`` volume, its frame count and its token grid; ``None`` when the
        layout does not fit the grid, e.g. when the tokens do not divide by it.
    """
    if grid is None or hidden <= 0:
        return None
    height, width = int(grid[0]), int(grid[1])
    cells = height * width
    if cells <= 0:
        return None
    if output.dim() == 3 and int(output.shape[-1]) == hidden:
        frames, remainder = divmod(int(output.shape[1]), cells)
        if remainder or frames < 1:
            return None
        volume = output.reshape(output.shape[0], frames, height, width, hidden)
        return volume.permute(0, 4, 1, 2, 3), frames, height, width
    if output.dim() == 4 and int(output.shape[-1]) == hidden and int(output.shape[2]) == cells:
        frames = int(output.shape[1])
        volume = output.reshape(output.shape[0], frames, height, width, hidden)
        return volume.permute(0, 4, 1, 2, 3), frames, height, width
    return None


def _denoiser_of(model: Any) -> Any:
    """The denoiser of a FlowWAM model or pipeline, or the object itself."""
    for name in ("denoiser", "dit", "backbone"):
        node = getattr(model, name, None)
        if isinstance(node, nn.Module):
            return node
    return model


def _registration(model: Any) -> Mapping[str, Any] | None:
    """The TIA registration kept on a model or on its denoiser, or ``None``."""
    for node in (model, _denoiser_of(model)):
        found = getattr(node, _REGISTRY_ATTR, None)
        if isinstance(found, Mapping):
            return found
    return None


def _token_grid(denoiser: Any) -> tuple[int, int] | None:
    """``tia_grid`` of a denoiser or its config as a ``(height, width)`` pair, or ``None``."""
    for source in (denoiser, getattr(denoiser, "config", None)):
        value = getattr(source, "tia_grid", None)
        if value is not None:
            return _as_grid(value)
    return None


def _block_count(denoiser: Any) -> int | None:
    """Number of blocks of a denoiser, or ``None`` when no container is exposed."""
    for name in BLOCK_CONTAINERS:
        container = getattr(denoiser, name, None)
        if isinstance(container, (nn.ModuleList, nn.Sequential, list, tuple)):
            return len(container)
    return None


def _normalise_state(state: Mapping[str, Any] | None) -> dict[str, torch.Tensor]:
    """Adapter weights under the projection names of this package, prefix and aliases removed."""
    if not state:
        return {}
    normalised: dict[str, torch.Tensor] = {}
    for key, value in dict(state).items():
        name = str(key)
        if name.startswith(_STATE_PREFIX):
            name = name[len(_STATE_PREFIX) :]
        name = _STATE_ALIASES.get(name, name)
        if name not in normalised:
            normalised[name] = torch.as_tensor(value)
    return normalised


def _load_state(adapter: TIAAdapter, state: Mapping[str, torch.Tensor]) -> None:
    """Load adapter weights, warning about every entry the adapter does not carry."""
    report = adapter.load_state_dict(dict(state), strict=False)
    unexpected = sorted(report.unexpected_keys)
    if unexpected:
        logger.warning(
            "Ignored %d TIA adapter entries this adapter does not carry: %s",
            len(unexpected),
            unexpected[:8],
        )
    if report.missing_keys:
        logger.debug("TIA adapter entries the checkpoint did not carry: %s", report.missing_keys)


def _hidden_size(
    *,
    state: Mapping[str, torch.Tensor],
    block: nn.Module,
    denoiser: Any,
) -> int | None:
    """Hidden size of a block, from an adapter checkpoint, the block or the denoiser."""
    for name, axis in _HIDDEN_KEYS:
        tensor = state.get(name)
        if tensor is not None and tensor.dim() > axis:
            return int(tensor.shape[axis])
    size = _module_hidden_size(block)
    if size is not None:
        return size
    size = _module_hidden_size(denoiser)
    if size is not None:
        return size
    dim = getattr(denoiser, "dim", None)
    if isinstance(dim, int) and dim > 0:
        return dim
    return None


def _rank(rank: int | None, state: Mapping[str, torch.Tensor]) -> int:
    """Low rank of a new adapter, from the argument, an adapter checkpoint or the defaults."""
    if rank is not None:
        return int(rank)
    for name, axis in _RANK_KEYS:
        tensor = state.get(name)
        if tensor is not None and tensor.dim() > axis:
            return int(tensor.shape[axis])
    return int(TIAConfig().dim)
