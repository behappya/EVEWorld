"""TIA adapter: the transport applied to one backbone block's hidden states.

The adapter is the inference-time half of the method. For every frame ``t >= 1`` it matches the
frame against its predecessor through the backbone's own patch correspondence
(:func:`eveworld.methods.tia.transport.transport`), subtracts the frame's own projection from the
transported one and feeds that delta through a **zero-initialised** output projection and a ``tanh``
squashing. Because ``W_out`` starts at zero the adapter is an exact identity when it is first
attached, so it can be dropped onto a released checkpoint and trained from there.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Any, Mapping

import torch
from torch import nn
from torch.utils.hooks import RemovableHandle

from eveworld.methods.tia.transport import transport

__all__ = ["TIAAdapter", "TIAConfig", "attach_adapter", "build_adapter", "remove_adapter"]


@dataclass
class TIAConfig:
    """Hyper-parameters of the TIA adapter (the paper's values are the defaults)."""

    dim: int = 64
    window: int = 7
    temperature: float = 0.07
    gamma: float = 0.1
    layer: int = 23
    first_frame_frozen: bool = True
    hidden_dim: int | None = None


def _coerce_config(config: TIAConfig | Mapping[str, Any] | None) -> TIAConfig:
    """Normalise the ``config`` argument of the public constructors."""
    if config is None:
        return TIAConfig()
    if isinstance(config, TIAConfig):
        return config
    if isinstance(config, Mapping):
        unknown = set(config) - {field.name for field in fields(TIAConfig)}
        if unknown:
            raise TypeError(f"unknown TIAConfig fields: {sorted(unknown)}")
        return TIAConfig(**dict(config))
    raise TypeError(f"config must be a TIAConfig, a mapping or None, got {type(config).__name__}")


class TIAAdapter(nn.Module):
    """Adjacent-frame local matching followed by a zero-init residual feedback.

    Args:
        hidden_size: Channel count of the block the adapter is attached to.
        config: Hyper-parameters, as a :class:`TIAConfig` or a mapping of its fields; ``None`` uses
            the paper's defaults. ``hidden_dim`` is filled in from ``hidden_size`` when unset.

    Raises:
        ValueError: If ``hidden_size`` does not match ``config.hidden_dim``, if ``dim``, ``window``,
        ``temperature`` or ``gamma`` are out of range.
        TypeError: If ``config`` is neither a :class:`TIAConfig`, a mapping nor ``None``.
    """

    def __init__(self, hidden_size: int, config: TIAConfig | Mapping[str, Any] | None = None) -> None:
        super().__init__()
        config = _coerce_config(config)
        hidden_size = int(hidden_size)
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        if config.hidden_dim is not None and int(config.hidden_dim) != hidden_size:
            raise ValueError(f"config hidden_dim {config.hidden_dim} does not match hidden_size {hidden_size}")
        if int(config.dim) <= 0:
            raise ValueError(f"dim must be positive, got {config.dim}")
        if int(config.window) < 1 or int(config.window) % 2 == 0:
            raise ValueError(f"window must be a positive odd size, got {config.window}")
        if float(config.temperature) <= 0:
            raise ValueError(f"temperature must be positive, got {config.temperature}")
        if float(config.gamma) < 0:
            raise ValueError(f"gamma must be non-negative, got {config.gamma}")

        self.config = replace(config, hidden_dim=hidden_size)
        self.hidden_size = hidden_size
        self.dim = int(config.dim)
        self.W_in = nn.Linear(hidden_size, self.dim, bias=False)
        self.W_out = nn.Linear(self.dim, hidden_size, bias=False)
        nn.init.zeros_(self.W_out.weight)
        self.hook_handle: RemovableHandle | None = None
        self.layer_index: int | None = None

    @property
    def window_radius(self) -> int:
        """Radius of the matching window, i.e. ``(window - 1) // 2``."""
        return (int(self.config.window) - 1) // 2

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, dim={self.dim}, "
            f"window={int(self.config.window)}x{int(self.config.window)}, "
            f"temperature={float(self.config.temperature)}, gamma={float(self.config.gamma)}, "
            f"first_frame_frozen={bool(self.config.first_frame_frozen)}"
        )

    def _as_clip(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int] | None, int]:
        """Flatten an accepted input layout to ``(B, T, N, C)`` plus its token grid and rank."""
        if hidden_states.ndim == 5:
            batch, channels, frames, height, width = hidden_states.shape
            if channels != self.hidden_size:
                raise ValueError(f"expected {self.hidden_size} channels, got {channels} for input {tuple(hidden_states.shape)}")
            clip = hidden_states.permute(0, 2, 3, 4, 1).reshape(batch, frames, height * width, channels)
            return clip, (int(height), int(width)), 5
        if hidden_states.ndim == 4:
            batch, frames, num_tokens, channels = hidden_states.shape
            if channels != self.hidden_size:
                raise ValueError(f"expected {self.hidden_size} channels, got {channels} for input {tuple(hidden_states.shape)}")
            return hidden_states, None, 4
        if hidden_states.ndim == 3:
            batch, num_tokens, channels = hidden_states.shape
            if channels != self.hidden_size:
                raise ValueError(f"expected {self.hidden_size} channels, got {channels} for input {tuple(hidden_states.shape)}")
            return hidden_states.unsqueeze(1), None, 3
        raise ValueError(f"expected (B, C, T, H, W), (B, T, N, C) or (B, N, C) input, got {tuple(hidden_states.shape)}")

    def _update(
        self,
        sources: torch.Tensor,
        donors: torch.Tensor,
        grid: tuple[int, int] | None,
    ) -> torch.Tensor:
        """Zero-init residual of ``gamma * tanh(W_out(transport(donor) - source))``.

        Args:
            sources: Float32 ``(B, T, N, C)`` projected frames that receive the update.
            donors: Float32 ``(B, T, N, C)`` projected donor frame of each source frame.
            grid: Optional ``(height, width)`` of the token grid.

        Returns:
            torch.Tensor: ``(B, T, N, C)`` residual in hidden-state space.
        """
        batch, frames, num_tokens, dim = sources.shape
        transported = transport(
            sources.reshape(batch * frames, num_tokens, dim),
            donors.reshape(batch * frames, num_tokens, dim),
            window=int(self.config.window),
            temperature=float(self.config.temperature),
            grid=grid,
        )
        delta = (transported - sources.reshape(batch * frames, num_tokens, dim)).reshape(sources.shape)
        return float(self.config.gamma) * torch.tanh(self.W_out(delta))

    def forward(self, hidden_states: torch.Tensor, frame_index: int | None = None) -> torch.Tensor:
        """Adapt a clip (or a single frame) of block hidden states.

        Args:
            hidden_states: ``(B, C, T, H, W)``, ``(B, T, N, C)`` or ``(B, N, C)`` block output. A
                single frame, or a clip with one frame, is returned unchanged because there is no
                neighbouring frame to transport from.
            frame_index: Temporal index of the first frame in ``hidden_states`` when the caller
                processes frames one at a time; ``None`` means the call carries the whole clip
                starting at frame 0. It only decides whether the frozen frame 0 is part of the call.

        Returns:
            torch.Tensor: Hidden states of the same shape and dtype. As long as ``W_out`` is zero
            the result is bit-identical to the input. With ``first_frame_frozen`` frame 0 is always
            returned unchanged; otherwise it borrows from frame 1 instead of from its predecessor.

        Raises:
            ValueError: If the input rank is unsupported or its channel count is not
            ``hidden_size``.
        """
        x = torch.as_tensor(hidden_states)
        clip, grid, ndim = self._as_clip(x)
        frames = clip.shape[1]
        if frames < 2:
            return x

        start = 0 if frame_index is None else int(frame_index)
        values = clip.to(torch.float32)
        projected = self.W_in(values)
        if self.config.first_frame_frozen and start == 0:
            donors = projected[:, :-1]
            update = self._update(projected[:, 1:], donors, grid)
            adapted = torch.cat((values[:, :1], values[:, 1:] + update), dim=1)
        else:
            donors = torch.cat((projected[:, 1:2], projected[:, :-1]), dim=1)
            adapted = values + self._update(projected, donors, grid)

        adapted = adapted.to(dtype=x.dtype)
        if ndim == 5:
            batch, frames = adapted.shape[0], adapted.shape[1]
            adapted = adapted.reshape(batch, frames, grid[0], grid[1], self.hidden_size).permute(0, 4, 1, 2, 3)
        elif ndim == 3:
            adapted = adapted.squeeze(1)
        return adapted


def build_adapter(
    hidden_size: int,
    config: TIAConfig | Mapping[str, Any] | None = None,
    **overrides: Any,
) -> TIAAdapter:
    """Build a :class:`TIAAdapter` from a config object, a mapping and keyword overrides.

    Args:
        hidden_size: Channel count of the block the adapter will be attached to.
        config: Base :class:`TIAConfig`, mapping of its fields, or ``None`` for the defaults.
        **overrides: Fields that take precedence over ``config``.

    Returns:
        TIAAdapter: The constructed adapter, with ``hidden_dim`` set to ``hidden_size``.

    Raises:
        ValueError: If ``hidden_size`` disagrees with a ``hidden_dim`` given in the config.
        TypeError: If ``config`` is of an unsupported type or an override is not a config field.
    """
    base = _coerce_config(config)
    if overrides:
        unknown = set(overrides) - {field.name for field in fields(TIAConfig)}
        if unknown:
            raise TypeError(f"unknown TIAConfig fields: {sorted(unknown)}")
        base = replace(base, **overrides)
    hidden_size = int(hidden_size)
    if base.hidden_dim is None:
        base = replace(base, hidden_dim=hidden_size)
    elif int(base.hidden_dim) != hidden_size:
        raise ValueError(f"config hidden_dim {base.hidden_dim} does not match hidden_size {hidden_size}")
    return TIAAdapter(hidden_size, base)


def _module_hidden_size(module: nn.Module) -> int | None:
    """Best-effort hidden size of a transformer block, or ``None`` when it is not exposed."""
    for name in ("hidden_size", "hidden_dim", "inner_dim", "n_embd", "model_channels", "out_features"):
        value = getattr(module, name, None)
        if value is None or isinstance(value, (torch.Tensor, nn.Module)):
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def attach_adapter(module: nn.Module, adapter: TIAAdapter, layer_index: int) -> TIAAdapter:
    """Attach ``adapter`` to the output of ``module`` through a forward hook.

    The hook adapts the block output in place (the returned value replaces the output of the
    module). Tuple outputs are supported: the first element is adapted and the remaining elements
    are passed through untouched, which matches the block signatures of the supported backbones.

    Args:
        module: The block whose output should be adapted.
        adapter: Adapter to attach; one already attached adapter is refused.
        layer_index: Temporal (block) index recorded on the adapter for reporting.

    Returns:
        TIAAdapter: The adapter, with ``hook_handle`` and ``layer_index`` set.

    Raises:
        TypeError: If ``adapter`` is not a :class:`TIAAdapter`.
        RuntimeError: If the adapter is already attached to a module.
        ValueError: If the block exposes a hidden size that differs from ``adapter.hidden_size``.
    """
    if not isinstance(adapter, TIAAdapter):
        raise TypeError(f"adapter must be a TIAAdapter, got {type(adapter).__name__}")
    if getattr(adapter, "hook_handle", None) is not None:
        raise RuntimeError(f"adapter is already attached at layer {getattr(adapter, 'layer_index', None)}; call remove_adapter first")
    expected = _module_hidden_size(module)
    if expected is not None and expected != adapter.hidden_size:
        raise ValueError(f"module hidden size {expected} does not match the adapter's {adapter.hidden_size}")

    def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> Any:
        if isinstance(output, tuple):
            return (adapter(output[0]), *output[1:])
        return adapter(output)

    adapter.hook_handle = module.register_forward_hook(hook)
    adapter.layer_index = int(layer_index)
    return adapter


def remove_adapter(adapter: TIAAdapter) -> TIAAdapter:
    """Detach an adapter attached by :func:`attach_adapter`.

    Args:
        adapter: The adapter to detach; detaching an unattached adapter is a no-op.

    Returns:
        TIAAdapter: The same adapter, with ``hook_handle`` and ``layer_index`` cleared.
    """
    handle = getattr(adapter, "hook_handle", None)
    if handle is not None:
        handle.remove()
        adapter.hook_handle = None
        adapter.layer_index = None
    return adapter
